# Project 3: WasteNet

## Background

Plastic pollution is one of the most pressing threats to our global oceans. From providing us with half of the oxygen we breathe and stabilizing our climate, to feeding over one billion people worldwide, the oceans are vital to our everyday wellbeing. However, each year an unprecedented amount of plastic litter- an average of 8 mln tonnes - make their way to the marine realm, having devastating consequences for wildlife and compromising our ecosystem's health. Because 80% of plastic litter is estimated to originate from land, we can all take action towards a healthier future for our environment and ourselves by tackling our individual plastic footprint. A 2018 study found that 1.1 billion of single-use items including bags and cups filled Metro Vancouver landfills - equivalent of 400 items per resident (source: TRI Environmental Consulting).

## Project Description

![](https://github.com/Vancouver-Datajam/project_3/blob/master/wastebin.png)

### Problem Statement

We at Vancouver Datajam are trying to predict the category for different waste items using image classification techniques in machine learning and computer vision. 

### Dataset

For our project, we work from a secondary dataset collected and maintained by [Gary Thung](https://github.com/garythung/trashnet) and [Mindy Yang](https://github.com/yangmindy4). The dataset consists of 2527 images of objects over six classes namely paper, glass, plastic, metal, cardboard, trash. Because we love trash so much, we also went out and took some more pictures using our phones while doing the hackathon! And also used some photos from [TACO](http://tacodataset.org/) for testing.

### Why is this interesting?

We think this problem is super interesting and relevant as it can help determine the category of recyclable items. 

## Team members: 
* Zubia Mansoor (Team Lead)
* Amir Parizi (Mentor)
* Seyed Mahdi Hosseini Miangoleh
* Saghar Irandoust
* Salina
* Warren Ho Kin
* Nasreen Mohsin
* Hans Sy
* Kevin Kang

## Datajam Schedule
| Time | Description |
| --- | --- |
| 8:30am | Opening Ceremony |
| 9:30am | Meet-up #1: Brainstorm and assign tasks|
| 10:00am | Optional Git workshop*|
| 10:30am | Hack & work on tasks |
| 1:30pm | Meet-up #2: Start forming the project|
| 3:30pm | Meet-up #3: Look at progress so far and finish up the project |
| 5:00pm | Final repository merging |
| 6:30pm | Project deadline & final Presentation! |
| 7:30pm | Career Panel & Q&A |
| 8:30pm | Awards Ceremony & Closing |


## How to use code in this repo

This repo contains training and testing code for our proposed model.

This script can be run either by using python notebook or python script file.(both are the same)

 ```python3 train.py --dataset_directory <the directory for the dataset here>```

Download the dataset from [here](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip), unzip the folder, and copy the text files inside the folder [list](https://github.com/zubiamansoor/project_3/tree/master/list) into the unzipped folder. 

To use your own dataset, you should have the images of different categories inside separate folders in the main dataset folder, and 3 seperate textfiles (test_list.txt, train_list.txt, val_list.txt) inside the main dataset folder, containing relative path for each image.

## Supplemental Materials

Team WasteNet accomplished a lot in ONE day and presented our findings at the end of the hackathon! You can find our slides [here.](https://github.com/zubiamansoor/project_3/blob/master/Presentation.pptx)

## References

1. [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385), Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun
