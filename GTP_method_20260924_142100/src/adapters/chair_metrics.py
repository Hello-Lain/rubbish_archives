"""SHIELD-compatible CHAIR scoring for the fixed COCO image subset."""

# The synonym table is copied from SHIELD's CHAIR evaluator and intentionally
# keeps its source line structure for auditability.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nltk

from utils.chair_singularize import singularize

SYNONYMS_TEXT = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_synonym_parser() -> tuple[set[str], dict[str, str], dict[str, str]]:
    groups = [
        [value.strip() for value in line.split(",") if value.strip()]
        for line in SYNONYMS_TEXT.splitlines()
        if line.strip()
    ]
    objects = {value for group in groups for value in group}
    inverse = {
        value: group[0]
        for group in groups
        for value in group
    }
    double_words = [
        "motor bike", "motor cycle", "air plane", "traffic light", "street light",
        "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
        "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
        "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
        "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
        "train track",
    ]
    double_word_dict = {word: word for word in double_words}
    for animal in [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "animal",
        "cub",
    ]:
        double_word_dict[f"baby {animal}"] = animal
        double_word_dict[f"adult {animal}"] = animal
    for vehicle in ["jet", "train"]:
        double_word_dict[f"passenger {vehicle}"] = vehicle
    double_word_dict["bow tie"] = "tie"
    double_word_dict["toilet seat"] = "toilet"
    double_word_dict["wine glas"] = "wine glass"
    return objects, inverse, double_word_dict


class ChairScorer:
    """Compute CHAIRs, CHAIRi, and macro grounded-object Recall."""

    def __init__(
        self,
        train_instances_path: Path,
        val_instances_path: Path,
        train_captions_path: Path,
        val_captions_path: Path,
        image_ids: list[int],
        cache_path: Path,
    ) -> None:
        self.train_instances_path = train_instances_path
        self.val_instances_path = val_instances_path
        self.train_captions_path = train_captions_path
        self.val_captions_path = val_captions_path
        self.image_ids = image_ids
        self.objects, self.inverse, self.double_word_dict = _build_synonym_parser()
        self.ground_truth = self._load_or_build_ground_truth(cache_path)

    def caption_to_words(
        self,
        caption: str,
    ) -> tuple[list[str], list[str], list[int], list[str]]:
        words = [
            singularize(word)
            for word in nltk.word_tokenize(caption.lower())
        ]
        merged_words: list[str] = []
        indexes: list[int] = []
        cursor = 0
        while cursor < len(words):
            double_word = " ".join(words[cursor : cursor + 2])
            if double_word in self.double_word_dict:
                merged_words.append(self.double_word_dict[double_word])
                indexes.append(cursor)
                cursor += 2
            else:
                merged_words.append(words[cursor])
                indexes.append(cursor)
                cursor += 1
        if "toilet" in merged_words and "seat" in merged_words:
            merged_words = [word for word in merged_words if word != "seat"]
        valid = [
            (word, indexes[index])
            for index, word in enumerate(merged_words)
            if word in self.objects
        ]
        valid_words = [word for word, _ in valid]
        valid_nodes = [self.inverse[word] for word in valid_words]
        valid_indexes = [index for _, index in valid]
        return valid_words, valid_nodes, valid_indexes, merged_words

    def score(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        sentence_rows: list[dict[str, Any]] = []
        hallucinated_captions = 0
        hallucinated_objects = 0
        mentioned_objects = 0
        recall_sum = 0.0
        for prediction in predictions:
            image_id = int(prediction["image_id"])
            caption = str(prediction["caption"])
            words, nodes, indexes, raw_words = self.caption_to_words(caption)
            ground_truth = set(self.ground_truth[image_id])
            hallucinated = [
                (word, node, index)
                for word, node, index in zip(words, nodes, indexes, strict=True)
                if node not in ground_truth
            ]
            grounded = {node for node in nodes if node in ground_truth}
            chair_s = int(bool(hallucinated))
            chair_i = len(hallucinated) / len(nodes) if nodes else 0.0
            recall = len(grounded) / len(ground_truth) if ground_truth else 0.0
            hallucinated_captions += chair_s
            hallucinated_objects += len(hallucinated)
            mentioned_objects += len(nodes)
            recall_sum += recall
            sentence_rows.append(
                {
                    "image_id": image_id,
                    "caption": caption,
                    "generated_objects": nodes,
                    "ground_truth_objects": sorted(ground_truth),
                    "hallucinated_objects": hallucinated,
                    "metrics": {
                        "CHAIRs": chair_s,
                        "CHAIRi": chair_i,
                        "Recall": recall,
                    },
                    "raw_words": raw_words,
                }
            )
        count = len(sentence_rows)
        return {
            "schema_version": "shield_chair_score_v2",
            "overall_metrics": {
                "CHAIRs": hallucinated_captions / count if count else 0.0,
                "CHAIRi": hallucinated_objects / mentioned_objects
                if mentioned_objects
                else 0.0,
                "Recall": recall_sum / count if count else 0.0,
                "hallucinated_objects": hallucinated_objects,
                "mentioned_objects": mentioned_objects,
            },
            "sentences": sentence_rows,
        }

    def _load_or_build_ground_truth(self, cache_path: Path) -> dict[int, set[str]]:
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = payload.get("ground_truth", {})
            image_ids = set(self.image_ids)
            if image_ids.issubset({int(key) for key in cached}):
                return {
                    image_id: set(cached[str(image_id)])
                    for image_id in self.image_ids
                }
        ground_truth: dict[int, set[str]] = defaultdict(set)
        for path in (self.train_instances_path, self.val_instances_path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            category_names = {
                int(category["id"]): self.inverse[category["name"]]
                for category in payload["categories"]
            }
            for annotation in payload["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in self.image_ids:
                    ground_truth[image_id].add(category_names[int(annotation["category_id"])])
        for path in (self.train_captions_path, self.val_captions_path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for annotation in payload["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in self.image_ids:
                    _, nodes, _, _ = self.caption_to_words(str(annotation["caption"]))
                    ground_truth[image_id].update(nodes)
        return {image_id: ground_truth[image_id] for image_id in self.image_ids}
