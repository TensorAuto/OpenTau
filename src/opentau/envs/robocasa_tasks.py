# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable integer indexing for the RoboCasa kitchen tasks (atomic + composite).

RoboCasa identifies tasks by registered class names (e.g. ``PnPCounterToCab``,
``MakeFruitBowl``). To match the LIBERO eval ergonomics — where a task suite plus integer
``task_ids`` selects which rollouts to run — we expose a frozen ordering here and resolve
``task_ids: [i, j, …]`` via :func:`resolve_task_ids`.

The registry is split into two append-only blocks:

* :data:`ROBOCASA_ATOMIC_TASKS` (indices ``0 .. 24``) — the 25 atomic kitchen skills from
  the RoboCasa paper (Nasiriany et al., 2024).
* :data:`ROBOCASA_COMPOSITE_TASKS` (indices ``25 .. 325``) — the 301 multi-stage
  composite kitchen tasks shipped under
  ``robocasa/environments/kitchen/composite/*`` upstream, ordered by category directory
  (alphabetical) and by file (alphabetical within each category) so that adjacent indices
  share a kitchen-skill family (e.g. all "baking" tasks are contiguous).

Both blocks together form a single flat index space, exposed as :data:`ROBOCASA_TASKS`.
The contract is **append-only**: existing indices must never be reassigned. Add new
tasks at the end of the appropriate block — never reorder, never delete.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# Atomic skills (indices 0-24). Names match the registered env class names in upstream
# robocasa (NOT the abbreviated labels from the paper — e.g. ``PickPlaceCounterToCabinet``
# rather than the paper's ``PnPCounterToCab``). Group order follows the RoboCasa paper's
# atomic-skill taxonomy so adjacent indices share a skill family — this makes things
# like ``task_ids: [0..7]`` (all pick-and-place tasks) intuitive without forcing every
# user to remember each class name.
ROBOCASA_ATOMIC_TASKS: tuple[str, ...] = (
    # Pick-and-Place (0-7)
    "PickPlaceCounterToCabinet",
    "PickPlaceCabinetToCounter",
    "PickPlaceCounterToSink",
    "PickPlaceSinkToCounter",
    "PickPlaceCounterToMicrowave",
    "PickPlaceMicrowaveToCounter",
    "PickPlaceCounterToStove",
    "PickPlaceStoveToCounter",
    # Opening / Closing Doors (8-11) — "single" → cabinet, "double" → fridge in upstream
    "OpenCabinet",
    "CloseCabinet",
    "OpenFridge",
    "CloseFridge",
    # Opening / Closing Drawers (12-13)
    "OpenDrawer",
    "CloseDrawer",
    # Twisting Knobs (14-15)
    "TurnOnStove",
    "TurnOffStove",
    # Turning Levers (16-18)
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    # Pressing Buttons (19-21) — coffee-machine "press" registers as StartCoffeeMachine
    "StartCoffeeMachine",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    # Insertion (22-23)
    "CoffeeServeMug",
    "CoffeeSetupMug",
    # Navigation (24)
    "NavigateKitchen",
)


# Composite multi-stage kitchen tasks (indices 25-325). Sourced from
# robocasa/environments/kitchen/composite/<category>/<task>.py upstream. Category order
# is alphabetical; within each category, files are listed alphabetically.
ROBOCASA_COMPOSITE_TASKS: tuple[str, ...] = (
    # Adding Ice To Beverages (25-28)
    "MakeIceLemonade",
    "PlaceEqualIceCubes",
    "PlaceIceInCup",
    "RetrieveIceTray",
    # Arranging Buffet (29-33)
    "ArrangeBuffetDessert",
    "CutBuffetPizza",
    "DivideBuffetTrays",
    "PlaceBeveragesTogether",
    "TongBuffetSetup",
    # Arranging Cabinets (34-36)
    "GatherTableware",
    "ResetCabinetDoors",
    "StackCans",
    # Arranging Condiments (37-39)
    "CategorizeCondiments",
    "LineUpCondiments",
    "OrganizeCondiments",
    # Baking (40-46)
    "CookieDoughPrep",
    "CoolBakedCake",
    "CoolBakedCookies",
    "CupcakeCleanup",
    "MixCakeFrosting",
    "OrganizeBakingIngredients",
    "PastryDisplay",
    # Boiling (47-54)
    "BoilCorn",
    "BoilEggs",
    "BoilPot",
    "CoolKettle",
    "FillKettle",
    "HeatMultipleWater",
    "PlaceLidToBoil",
    "StartElectricKettle",
    # Brewing (55-60)
    "ArrangeTea",
    "DeliverBrewedCoffee",
    "KettleBoiling",
    "OrganizeCoffeeCondiments",
    "PrepareCoffee",
    "SweetenCoffee",
    # Broiling Fish (61-65)
    "OvenBroilFish",
    "PrepareBroilingStation",
    "RemoveBroiledFish",
    "ToasterOvenBroilFish",
    "WashFish",
    # Chopping Food (66-71)
    "ArrangeCuttingFruits",
    "ArrangeVegetables",
    "BreadSetupSlicing",
    "ClearCuttingBoard",
    "MeatTransfer",
    "OrganizeVegetables",
    # Chopping Vegetables (72-73)
    "CuttingToolSelection",
    "GatherCuttingTools",
    # Cleaning Appliances (74-76)
    "CleanBlenderJug",
    "PrepFridgeForCleaning",
    "PrepSinkForCleaning",
    # Cleaning Sink (77-79)
    "ClearFoodWaste",
    "ClearSinkArea",
    "RinseSinkBasin",
    # Clearing Table (80-87)
    "BowlAndCup",
    "CandleCleanup",
    "ClearReceptaclesForCleaning",
    "ClusterItemsForClearing",
    "CondimentCollection",
    "DessertAssembly",
    "DrinkwareConsolidation",
    "FoodCleanup",
    # Defrosting Food (88-93)
    "DefrostByCategory",
    "MicrowaveThawing",
    "MicrowaveThawingFridge",
    "MoveToCounter",
    "QuickThaw",
    "ThawInSink",
    # Filling Serving Dishes (94-97)
    "BuildAppetizerPlate",
    "DisplayMeatVariety",
    "MeatSkewerAssembly",
    "MixedFruitPlatter",
    # Frying (98-105)
    "AssembleCookingArray",
    "DistributeSteakOnPans",
    "FryingPanAdjustment",
    "MealPrepStaging",
    "PressChicken",
    "RotatePan",
    "SearingMeat",
    "SetupFrying",
    # Garnishing Dishes (106-110)
    "AddLemonToFish",
    "AddSugarCubes",
    "GarnishCake",
    "GarnishCupcake",
    "GarnishPancake",
    # Loading Dishwasher (111-112)
    "LoadDishwasher",
    "PrepareDishwasher",
    # Loading Fridge (113-120)
    "CreateChildFriendlyFridge",
    "LoadCondimentsInFridge",
    "LoadFridgeByType",
    "LoadFridgeFifo",
    "LoadPreparedFood",
    "MoveFreezerToFridge",
    "PlaceVeggiesInDrawer",
    "RearrangeFridgeItems",
    # Making Juice (121-123)
    "ChooseRipeFruit",
    "FillBlenderJug",
    "JuiceFruitReamer",
    # Making Salads (124-125)
    "PrepareCheeseStation",
    "WashLettuce",
    # Making Smoothies (126-130)
    "AddIceCubes",
    "AddSweetener",
    "BlendIngredients",
    "PlaceStraw",
    "PrepareSmoothie",
    # Making Tea (131-133)
    "ArrangeTeaAccompaniments",
    "ServeTea",
    "StrainerSetup",
    # Making Toast (134-136)
    "BreadSelection",
    "PrepareToast",
    "SweetSavoryToastSetup",
    # Managing Freezer Space (137-144)
    "ClearFreezer",
    "FreezeBottledWaters",
    "FreezeIceTray",
    "MaximizeFreezerSpace",
    "MoveFridgeToFreezer",
    "MoveToFreezerDrawer",
    "ReorganizeFrozenVegetables",
    "SeparateFreezerRack",
    # Measuring Ingredients (145-147)
    "ChooseMeasuringCup",
    "OrganizeMeasuringCups",
    "WeighIngredients",
    # Meat Preparation (148-149)
    "PrepForTenderizing",
    "PrepMarinatingMeat",
    # Microwaving Food (150-155)
    "FilterMicrowavableItem",
    "MicrowaveCorrectMeal",
    "MicrowaveDefrostMeat",
    "PlaceMicrowaveSafeItem",
    "ReheatMeal",
    "ReturnHeatedFood",
    # Mixing And Blending (156-158)
    "ColorfulSalsa",
    "MakeBananaMilkshake",
    "SpicyMarinade",
    # Mixing Ingredients (159-164)
    "BlendSalsaMix",
    "BlendVegetableSauce",
    "CheeseMixing",
    "MakeCheesecakeFilling",
    "MakeChocolateMilk",
    "PrepareVeggieDip",
    # Organizing Dishes And Containers (165-167)
    "EmptyDishRack",
    "OrganizeMugsByHandle",
    "StackBowlsCabinet",
    # Organizing Recycling (168-171)
    "RecycleBottlesBySize",
    "RecycleBottlesByType",
    "RecycleSodaCans",
    "RecycleStackedYogurt",
    # Organizing Utensils (172-174)
    "ArrangeUtensilsByType",
    "ClusterUtensilsInDrawer",
    "OrganizeMetalicUtensils",
    # Packing Lunches (175-178)
    "PackFoodByTemp",
    "PackFruitContainer",
    "PackIdenticalLunches",
    "PackSnack",
    # Plating Food (179-181)
    "BalancedMealPrep",
    "PlateSteakMeal",
    "PlateStoreDinner",
    # Portioning Meals (182-188)
    "DistributeChicken",
    "PortionFruitBowl",
    "PortionHotDogs",
    "PortionInTupperware",
    "PortionOnSize",
    "PortionYogurt",
    "ScalePortioning",
    # Preparing Hot Chocolate (189-190)
    "AddMarshmallow",
    "SweetenHotChocolate",
    # Preparing Marinade (191-193)
    "BlendMarinade",
    "GatherMarinadeIngredients",
    "PlaceMeatInMarinade",
    # Preparing Sandwiches (194-199)
    "GatherVegetables",
    "HeatKebabSandwich",
    "HotDogSetup",
    "PrepareSandwichStation",
    "PrepareSausageCheese",
    "ToastHeatableIngredients",
    # Reheating Food (200-205)
    "HeatMug",
    "MakeLoadedPotato",
    "ReheatMeatOnStove",
    "SimmeringSauce",
    "WaffleReheat",
    "WarmCroissant",
    # Restocking Supplies (206-213)
    "BeverageSorting",
    "FreshProduceOrganization",
    "RefillCondimentStation",
    "RestockBowls",
    "RestockCannedFood",
    "RestockPantry",
    "RestockSinkSupplies",
    "StockingBreakfastFoods",
    # Sanitizing Cutting Board (214-217)
    "RemoveCuttingBoardItems",
    "RinseCuttingBoard",
    "SanitizePrepCuttingBoard",
    "ScrubCuttingBoard",
    # Sanitizing Surface (218-223)
    "ArrangeSinkSanitization",
    "CleanMicrowave",
    "CountertopCleanup",
    "PrepForSanitizing",
    "SanitizeSink",
    "WipeTable",
    # Sauteing Vegetables (224-230)
    "AdjustHeat",
    "ButterOnPan",
    "PlaceVegetablesEvenly",
    "PreheatPot",
    "ShakePan",
    "StirVegetables",
    "TiltPan",
    # Seasoning Food (231-233)
    "LemonSeasoningFish",
    "SeasoningSteak",
    "SetupSpiceStation",
    # Serving Beverages (234-239)
    "DeliverStraw",
    "MatchCupAndDrink",
    "PrepareCocktailStation",
    "PrepareDrinkStation",
    "ServeMealJuice",
    "SetupSodaBowl",
    # Serving Food (240-245)
    "AlcoholServingPrep",
    "DessertUpgrade",
    "PanTransfer",
    "PlaceFoodInBowls",
    "PrepareSoupServing",
    "ServeSteak",
    # Setting The Table (246-258)
    "AlignSilverware",
    "ArrangeBreadBasket",
    "ArrangeBreadBowl",
    "ArrangeDrinkware",
    "BeverageOrganization",
    "DateNight",
    "SeasoningSpiceSetup",
    "SetBowlsForSoup",
    "SetupBowls",
    "SetupButterPlate",
    "SetupFruitBowl",
    "SetupWineGlasses",
    "SizeSorting",
    # Simmering Sauces (259)
    "TurnOffSimmeredSauceHeat",
    # Slicing Meat (260-262)
    "CleanBoard",
    "RetrieveMeat",
    "SetUpCuttingStation",
    # Slow Cooking (263-265)
    "AddToSoupPot",
    "BeginSlowCooking",
    "StopSlowCooking",
    # Snack Preparation (266-270)
    "BreadAndCheese",
    "CerealAndBowl",
    "MakeFruitBowl",
    "VeggieDipPrep",
    "YogurtDelightPrep",
    # Sorting Ingredients (271-272)
    "SeparateRawIngredients",
    "SortBreakfastIngredients",
    # Steaming Food (273-275)
    "MultistepSteaming",
    "SteamFish",
    "SteamInMicrowave",
    # Steaming Vegetables (276-278)
    "PrepareVeggiesForSteaming",
    "RemoveSteamedVegetables",
    "SteamVeggiesWithWater",
    # Storing Leftovers (279-283)
    "FreezeCookedFood",
    "PrepareStoringLeftovers",
    "StoreDumplings",
    "StoreLeftoversByType",
    "StoreLeftoversInBowl",
    # Tidying Cabinets And Drawers (284-288)
    "DrawerUtensilSort",
    "OrganizeCleaningSupplies",
    "PlaceBreakfastItemsAway",
    "SnackSorting",
    "UtensilShuffle",
    # Toasting Bread (289-295)
    "GetToastedBread",
    "PjSandwichPrep",
    "ServeWarmCroissant",
    "ToastBagel",
    "ToastBaguette",
    "ToastOnCorrectRack",
    "ToastOneSlotPair",
    # Washing Dishes (296-314)
    "ChangeWaterTemp",
    "ClearSink",
    "CollectWashingSupplies",
    "DivideBasins",
    "DryDishes",
    "DryDrinkware",
    "DumpLeftovers",
    "PlaceDishesBySink",
    "PlaceOnDishRack",
    "PreRinseStation",
    "PreSoakPan",
    "ReturnWashingSupplies",
    "RinseBowls",
    "RinseFragileItem",
    "ScrubBowl",
    "SoakSponge",
    "SortingCleanup",
    "StackBowls",
    "TransportCookware",
    # Washing Fruits And Vegetables (315-325)
    "AfterwashSorting",
    "AirdryFruit",
    "ClearClutter",
    "ClearSinkSpace",
    "DrainVeggies",
    "GatherProduceWashing",
    "PrepareVegetableRoasting",
    "PrewashFoodAssembly",
    "PrewashFoodSorting",
    "WashFruitColander",
    "WashInSaucepan",
)


# Flat lookup table: indices 0..len(ROBOCASA_ATOMIC_TASKS)-1 are atomic, the rest are
# composite. Concatenation order is fixed and must never be reordered.
ROBOCASA_TASKS: tuple[str, ...] = ROBOCASA_ATOMIC_TASKS + ROBOCASA_COMPOSITE_TASKS

# Sanity check: no duplicate class names in the registry. A duplicate would make
# ``task_index`` non-deterministic and silently break round-trips.
assert len(set(ROBOCASA_TASKS)) == len(ROBOCASA_TASKS), (
    "Duplicate RoboCasa task names in registry — fix in robocasa_tasks.py."
)

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ROBOCASA_TASKS)}
_ATOMIC_END: int = len(ROBOCASA_ATOMIC_TASKS)


def task_name(task_id: int) -> str:
    """Return the RoboCasa task class name for ``task_id``.

    Raises:
        ValueError: ``task_id`` is out of range for :data:`ROBOCASA_TASKS`.
    """
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise TypeError(f"task_id must be an int, got {type(task_id).__name__}")
    if task_id < 0 or task_id >= len(ROBOCASA_TASKS):
        raise ValueError(
            f"RoboCasa task_id {task_id} out of range [0, {len(ROBOCASA_TASKS) - 1}]. "
            f"See ROBOCASA_TASKS in opentau.envs.robocasa_tasks for the full list "
            f"(atomic: 0-{_ATOMIC_END - 1}, composite: {_ATOMIC_END}-{len(ROBOCASA_TASKS) - 1})."
        )
    return ROBOCASA_TASKS[task_id]


def task_index(name: str) -> int:
    """Inverse of :func:`task_name` — return the integer index for a task class name."""
    try:
        return _NAME_TO_INDEX[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown RoboCasa task name {name!r}. "
            f"Known atomic: {', '.join(ROBOCASA_ATOMIC_TASKS)}. "
            f"({len(ROBOCASA_COMPOSITE_TASKS)} composite tasks omitted — see "
            f"ROBOCASA_COMPOSITE_TASKS in opentau.envs.robocasa_tasks.)"
        ) from e


def is_atomic(task_id_or_name: int | str) -> bool:
    """Return True iff the given task is one of the atomic kitchen skills."""
    if isinstance(task_id_or_name, str):
        return task_id_or_name in ROBOCASA_ATOMIC_TASKS
    return 0 <= task_id_or_name < _ATOMIC_END


def is_composite(task_id_or_name: int | str) -> bool:
    """Return True iff the given task is one of the composite multi-stage tasks."""
    if isinstance(task_id_or_name, str):
        return task_id_or_name in ROBOCASA_COMPOSITE_TASKS
    return _ATOMIC_END <= task_id_or_name < len(ROBOCASA_TASKS)


def resolve_task_ids(task_ids: Iterable[int]) -> list[str]:
    """Map an iterable of integer task IDs to a list of RoboCasa task class names.

    Order is preserved; duplicates are kept (the caller decides whether to dedupe).
    """
    return [task_name(int(tid)) for tid in task_ids]


def resolve_tasks(
    task: str | Sequence[str] | None,
    task_ids: Iterable[int] | None,
) -> list[str]:
    """Normalise a ``task`` / ``task_ids`` pair into a flat list of task class names.

    Accepts either:
      * ``task``: comma-separated class names (e.g. ``"PnPCounterToCab,MakeFruitBowl"``)
        or a sequence of names, AND/OR
      * ``task_ids``: a list of integer indices into :data:`ROBOCASA_TASKS` (atomic in
        ``0 .. 24``, composite in ``25 .. 325``).

    Names from both sources are concatenated (``task`` first, then ``task_ids``);
    duplicates are preserved so callers can stack repeats deliberately. At least one of
    the two arguments must yield a non-empty list.
    """
    names: list[str] = []
    if task:
        if isinstance(task, str):
            names.extend(s.strip() for s in task.split(",") if s.strip())
        else:
            names.extend(str(t).strip() for t in task if str(t).strip())
    if task_ids:
        names.extend(resolve_task_ids(task_ids))
    if not names:
        raise ValueError(
            "RoboCasa task resolution produced an empty list — provide `task` (class "
            "names) and/or `task_ids` (indices into ROBOCASA_TASKS)."
        )
    return names
