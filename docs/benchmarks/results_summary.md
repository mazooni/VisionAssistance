## Shared-class comparison (classes all models can predict)

| Benchmark | Model | mAP50-95 | mAP50 | classes |
|---|---|---|---|---|
| cocoval (4998 imgs) | ours_v3 | 0.379 | 0.567 | 16 |
| cocoval (4998 imgs) | yolo26n | 0.434 | 0.601 | 16 |
| cocoval (4998 imgs) | yolo12n | 0.434 | 0.597 | 16 |
| wotr (1594 imgs) | ours_v3 | 0.295 | 0.497 | 9 |
| wotr (1594 imgs) | yolo26n | 0.278 | 0.451 | 9 |
| wotr (1594 imgs) | yolo12n | 0.274 | 0.452 | 9 |
| mapillary (2000 imgs) | ours_v3 | 0.123 | 0.210 | 10 |
| mapillary (2000 imgs) | yolo26n | 0.120 | 0.203 | 10 |
| mapillary (2000 imgs) | yolo12n | 0.123 | 0.203 | 10 |

## Custom classes (only our model can predict these)

| Benchmark | mAP50-95 | mAP50 | classes |
|---|---|---|---|
| wotr | 0.208 | 0.349 | pole, spherical_roadblock, traffic_cone, trash_can, tricycle, warning_column |
| mapillary | 0.033 | 0.066 | pole, traffic_cone, trash_can |
| door | 0.679 | 0.865 | door |
| stairs | 0.493 | 0.794 | stairs |

## Per-class AP50 across models


### cocoval

| class | ours_v3 AP50 | yolo26n AP50 | yolo12n AP50 | ours P@.25 | ours R@.25 | n_gt |
|---|---|---|---|---|---|---|
| train | 0.858 | 0.884 | 0.853 | 0.8698 | 0.7737 | 190 |
| fire_hydrant | 0.781 | 0.808 | 0.789 | 0.8415 | 0.69 | 100 |
| bus | 0.761 | 0.775 | 0.761 | 0.824 | 0.6784 | 283 |
| person | 0.759 | 0.764 | 0.773 | 0.8246 | 0.6164 | 10773 |
| stop_sign | 0.683 | 0.680 | 0.695 | 0.8302 | 0.5867 | 75 |
| motorcycle | 0.682 | 0.667 | 0.692 | 0.808 | 0.5504 | 367 |
| car | 0.580 | 0.575 | 0.584 | 0.7517 | 0.4713 | 1914 |
| umbrella | 0.559 | 0.567 | 0.591 | 0.681 | 0.4668 | 407 |
| dog | 0.559 | 0.778 | 0.756 | 0.4762 | 0.6881 | 218 |
| bicycle | 0.490 | 0.491 | 0.481 | 0.7595 | 0.3822 | 314 |
| truck | 0.470 | 0.483 | 0.408 | 0.6465 | 0.3366 | 413 |
| suitcase | 0.465 | 0.561 | 0.559 | 0.5833 | 0.398 | 299 |
| traffic_light | 0.441 | 0.427 | 0.422 | 0.7228 | 0.328 | 628 |
| chair | 0.383 | 0.428 | 0.437 | 0.6616 | 0.2682 | 1771 |
| bench | 0.300 | 0.318 | 0.333 | 0.5787 | 0.2506 | 411 |
| potted_plant | 0.299 | 0.406 | 0.417 | 0.4933 | 0.3246 | 342 |

### wotr

| class | ours_v3 AP50 | yolo26n AP50 | yolo12n AP50 | ours P@.25 | ours R@.25 | n_gt |
|---|---|---|---|---|---|---|
| car | 0.754 | 0.732 | 0.744 | 0.8563 | 0.5894 | 3741 |
| person | 0.635 | 0.600 | 0.626 | 0.8259 | 0.4752 | 4853 |
| traffic_light | 0.543 | 0.494 | 0.533 | 0.6797 | 0.4373 | 1834 |
| bus | 0.526 | 0.517 | 0.471 | 0.7788 | 0.4112 | 214 |
| motorcycle | 0.525 | 0.451 | 0.451 | 0.8652 | 0.2901 | 2389 |
| tricycle | 0.479 | — | — | 0.9524 | 0.3529 | 170 |
| dog | 0.472 | 0.426 | 0.396 | 1.0 | 0.25 | 16 |
| traffic_cone | 0.459 | — | — | 0.9045 | 0.3205 | 443 |
| trash_can | 0.443 | — | — | 0.8981 | 0.2594 | 374 |
| truck | 0.364 | 0.269 | 0.272 | 0.5429 | 0.3287 | 289 |
| fire_hydrant | 0.347 | 0.270 | 0.236 | 0.75 | 0.253 | 83 |
| spherical_roadblock | 0.311 | — | — | 0.9313 | 0.2373 | 628 |
| bicycle | 0.308 | 0.304 | 0.341 | 0.6543 | 0.2032 | 1127 |
| warning_column | 0.258 | — | — | 0.7736 | 0.1272 | 1801 |
| pole | 0.143 | — | — | 0.6423 | 0.0313 | 5339 |

### mapillary

| class | ours_v3 AP50 | yolo26n AP50 | yolo12n AP50 | ours P@.25 | ours R@.25 | n_gt |
|---|---|---|---|---|---|---|
| car | 0.498 | 0.486 | 0.480 | 0.8091 | 0.3835 | 14578 |
| person | 0.332 | 0.315 | 0.314 | 0.8317 | 0.2235 | 6855 |
| bus | 0.258 | 0.251 | 0.247 | 0.5893 | 0.219 | 452 |
| traffic_light | 0.246 | 0.240 | 0.227 | 0.823 | 0.15 | 7626 |
| motorcycle | 0.202 | 0.183 | 0.182 | 0.7685 | 0.1263 | 657 |
| bicycle | 0.174 | 0.184 | 0.167 | 0.757 | 0.1149 | 705 |
| fire_hydrant | 0.132 | 0.134 | 0.116 | 0.6552 | 0.0892 | 213 |
| truck | 0.128 | 0.115 | 0.140 | 0.2093 | 0.1742 | 775 |
| traffic_cone | 0.122 | — | — | 0.9333 | 0.0224 | 625 |
| train | 0.085 | 0.076 | 0.106 | 0.2667 | 0.1 | 40 |
| bench | 0.049 | 0.049 | 0.050 | 0.3333 | 0.0357 | 224 |
| trash_can | 0.046 | — | — | 0.0 | 0.0 | 783 |
| pole | 0.029 | — | — | 0.5796 | 0.0093 | 38136 |

### door

| class | ours_v3 AP50 | yolo26n AP50 | yolo12n AP50 | ours P@.25 | ours R@.25 | n_gt |
|---|---|---|---|---|---|---|
| door | 0.865 | — | — | 0.84 | 0.7778 | 135 |

### stairs

| class | ours_v3 AP50 | yolo26n AP50 | yolo12n AP50 | ours P@.25 | ours R@.25 | n_gt |
|---|---|---|---|---|---|---|
| stairs | 0.794 | — | — | 0.7936 | 0.7309 | 747 |


# Size-stratified results (relative box area; virtual-1000px COCOeval)

Bins: tiny <0.1% of image area · small 0.1-1% · medium 1-5% · large >5%.
Motivation: assist-app operating envelope is medium/large; headline mAP is
dominated by tiny background instances (WOTR: 84% of GT <1%; Mapillary: 65% tiny).


## cocoval

| model | imgsz | block | tiny | small | medium | large | all |
|---|---|---|---|---|---|---|---|
| ours_v3 | 640 | shared | 0.1208 | 0.4113 | 0.6267 | 0.7559 | 0.5669 |
| yolo12n | 640 | shared | 0.0866 | 0.4525 | 0.6511 | 0.7963 | 0.5969 |
| yolo26n | 640 | shared | 0.121 | 0.4218 | 0.6585 | 0.8058 | 0.6007 |

GT bin counts: {'tiny': 2442, 'small': 6244, 'medium': 4662, 'large': 5157, 'all': 18505}

## wotr

| model | imgsz | block | tiny | small | medium | large | all |
|---|---|---|---|---|---|---|---|
| ours_v3 | 640 | shared | 0.2063 | 0.4884 | 0.8079 | 0.9022 | 0.4972 |
| ours_v3 | 640 | custom | 0.092 | 0.3138 | 0.6177 | 0.8927 | 0.3488 |
| yolo12n | 640 | shared | 0.1521 | 0.4618 | 0.7766 | 0.7693 | 0.452 |
| yolo26n | 640 | shared | 0.1492 | 0.4435 | 0.8032 | 0.8456 | 0.4514 |

GT bin counts: {'tiny': 6854, 'small': 12889, 'medium': 2637, 'large': 921, 'all': 23301}

## mapillary

| model | imgsz | block | tiny | small | medium | large | all |
|---|---|---|---|---|---|---|---|
| ours_v3 | 640 | shared | 0.0634 | 0.3604 | 0.6417 | 0.76 | 0.2103 |
| ours_v3 | 640 | custom | 0.0202 | 0.2135 | 0.4114 | 0.1883 | 0.0656 |
| ours_v3_1280 | 1280 | shared | 0.1559 | 0.5085 | 0.6913 | 0.7474 | 0.3244 |
| ours_v3_1280 | 1280 | custom | 0.0711 | 0.2898 | 0.4283 | 0.0799 | 0.1182 |
| yolo12n | 640 | shared | 0.0494 | 0.3586 | 0.661 | 0.767 | 0.2029 |
| yolo26n | 640 | shared | 0.0583 | 0.3438 | 0.6248 | 0.7749 | 0.2032 |

GT bin counts: {'tiny': 46810, 'small': 18763, 'medium': 4811, 'large': 1285, 'all': 71669}
