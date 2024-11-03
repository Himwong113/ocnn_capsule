import json
import pandas as pd


catergory ={
"airplane":[0],
"bathtub":[0],
"bed":[0],
"bench":[0],
"bookshelf":[0],
"bottle":[0],
"bowl":[0],
"car":[0],
"chair":[0],
"cone":[0],
"cup":[0],
"curtain":[0],
"desk":[0],
"door":[0],
"dresser":[0],
"flower_pot":[0],
"glass_box":[0],
"guitar":[0],
"keyboard":[0],
"lamp":[0],
"laptop":[0],
"mantel":[0],
"monitor":[0],
"night_stand":[0],
"person":[0],
"piano":[0],
"plant":[0],
"radio":[0],
"range_hood":[0],
"sink":[0],
"sofa":[0],
"stairs":[0],
"stool":[0],
"table":[0],
"tent":[0],
"toilet":[0],
"tv_stand":[0],
"vase":[0],
"wardrobe":[0],
"xbox":[0],
}

catergory_train =catergory
catergory_test =catergory


for i in range(5):
    j = open(f'ply_data_train_{i}_id2file.json')

    for line in json.load(j):

        catergory_train [ line.split('/')[0]][0]+=1


catergory ={
"airplane":[0],
"bathtub":[0],
"bed":[0],
"bench":[0],
"bookshelf":[0],
"bottle":[0],
"bowl":[0],
"car":[0],
"chair":[0],
"cone":[0],
"cup":[0],
"curtain":[0],
"desk":[0],
"door":[0],
"dresser":[0],
"flower_pot":[0],
"glass_box":[0],
"guitar":[0],
"keyboard":[0],
"lamp":[0],
"laptop":[0],
"mantel":[0],
"monitor":[0],
"night_stand":[0],
"person":[0],
"piano":[0],
"plant":[0],
"radio":[0],
"range_hood":[0],
"sink":[0],
"sofa":[0],
"stairs":[0],
"stool":[0],
"table":[0],
"tent":[0],
"toilet":[0],
"tv_stand":[0],
"vase":[0],
"wardrobe":[0],
"xbox":[0],
}
catergory_test =catergory
for i in range(2):
    j = open(f'ply_data_test_{i}_id2file.json')

    for line in json.load(j):
        catergory_test [ line.split('/')[0]][0]+=1

summary ={'train':catergory_train,'test':catergory_test}

df =pd.DataFrame(summary)
df.to_csv('train_cat_summary.csv')