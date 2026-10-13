# 내가 TIFA 점수를 얻고 싶은것은
#

from tifascore import tifa_score_benchmark
import json

# We recommend using mplug-large
results = tifa_score_benchmark("mplug-large", "sample/sample_question_answers.json", "sample/sample_imgs.json")

# save the results
with open("sample/sample_evaluation_result.json", "w") as f:
    json.dump(results, f, indent=4)

########################################################################################################
# sample/sample_question_answers.json
# [
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is this a surfer?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "animal/human",
#         "element": "surfer",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "who is on the beach?",
#         "choices": [
#             "surfer",
#             "swimmer",
#             "diver",
#             "skier"
#         ],
#         "answer": "surfer",
#         "element_type": "animal/human",
#         "element": "surfer",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is the person carrying a board?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "object",
#         "element": "board",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is this a beach?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "location",
#         "element": "beach",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "what type of place is this?",
#         "choices": [
#             "beach",
#             "park",
#             "forest",
#             "mountain"
#         ],
#         "answer": "beach",
#         "element_type": "location",
#         "element": "beach",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is the surfer carrying something?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "activity",
#         "element": "carrying",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is the surfer walking on the beach?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "activity",
#         "element": "walks",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is the day gray?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "color",
#         "element": "gray",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "what color is the sky?",
#         "choices": [
#             "gray",
#             "black",
#             "red",
#             "blue"
#         ],
#         "answer": "gray",
#         "element_type": "color",
#         "element": "gray",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "is the board white?",
#         "choices": [
#             "yes",
#             "no"
#         ],
#         "answer": "yes",
#         "element_type": "color",
#         "element": "white",
#         "coco_val_id": "380711"
#     },
#     {
#         "id": "coco_301091",
#         "caption": "On a gray day a surfer carrying a white board walks on a beach.",
#         "question": "what color is the board?",
#         "choices": [
#             "white",
#             "black",
#             "red",
#             "blue"
#         ],
#         "answer": "white",
#         "element_type": "color",
#         "element": "white",
#         "coco_val_id": "380711"
#     },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "are there cats?",
#             "choices": [
#                 "yes",
#                 "no"
#             ],
#             "answer": "yes",
#             "element_type": "animal/human",
#             "element": "cats"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "are there dogs?",
#             "choices": [
#                 "yes",
#                 "no"
#             ],
#             "answer": "yes",
#             "element_type": "animal/human",
#             "element": "dogs"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "is there grass?",
#             "choices": [
#                 "yes",
#                 "no"
#             ],
#             "answer": "yes",
#             "element_type": "object",
#             "element": "grass"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "what is the animals sitting on?",
#             "choices": [
#                 "grass",
#                 "leaves",
#                 "twigs",
#                 "sand"
#             ],
#             "answer": "grass",
#             "element_type": "object",
#             "element": "grass"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "are the animals sitting?",
#             "choices": [
#                 "yes",
#                 "no"
#             ],
#             "answer": "yes",
#             "element_type": "activity",
#             "element": "sitting"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "how many cats are in the picture?",
#             "choices": [
#                 "1",
#                 "2",
#                 "3",
#                 "4"
#             ],
#             "answer": "3",
#             "element_type": "counting",
#             "element": "three"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "are there two dogs?",
#             "choices": [
#                 "yes",
#                 "no"
#             ],
#             "answer": "yes",
#             "element_type": "counting",
#             "element": "two"
#         },
#         {
#             "id": "drawbench_52",
#             "caption": "Three cats and two dogs sitting on the grass.",
#             "question": "how many dogs are in the picture?",
#             "choices": [
#                 "1",
#                 "2",
#                 "3",
#                 "4"
#             ],
#             "answer": "2",
#             "element_type": "counting",
#             "element": "two"
#         }
# ]
########################################################################################################
# sample/sample_imgs.json
# {
#     "coco_301091": "coco_301091.jpg",
#     "drawbench_52": "drawbench_52.jpg"
# }
