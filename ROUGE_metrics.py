import os
from rouge import Rouge
import string
import pandas as pd

gold_list = []
gold_path = "/path-to/gold_summary"
auto_path = "/path-to/auto-summary"
#gold summaries located in one folder 
for file in os.listdir(gold_path):
    if os.path.isfile(os.path.join(gold_path, file)):
        gold_list.append(file)
#auto summaries located in one folder 
auto_list = []
for file in os.listdir(auto_path):
    if os.path.isfile(os.path.join(auto_path, file)):
        auto_list.append(file)

print(f"Gold List: {gold_list}")
print(f"Auto List: {auto_list}")

scores = {}
rouge = Rouge()  # Instantiate ROUGE outside the loop to avoid redundancy

for ite in gold_list:
    for item in auto_list:
        if ite == item:  
            with open(os.path.join(auto_path, item), 'r') as f1:
                a = f1.read().translate(str.maketrans('', '', string.punctuation))
            model = a.lower()

            with open(os.path.join(gold_path, ite), 'r') as f2:
                gold_read = f2.read().translate(str.maketrans('', '', string.punctuation))
            gold = gold_read.lower()

            name = f"{item}_and_{ite}"
            scores[name] = rouge.get_scores(model, gold)
            print(f"ROUGE Scores for {item}: {scores[name]}")

# Initialize lists for storing F-scores
R1_f_score = []
R2_f_score = []
RL_f_score = []

#find rouge f-score
for ke, val in scores.items():
    for it in val:
        for k1, v1 in it.items():
            if k1 == 'rouge-1':
                R1_f = v1['f'] 
                R1_f_score.append(R1_f)
            elif k1 == 'rouge-2':
                R2_f = v1['f']
                R2_f_score.append(R2_f)
            elif k1 == 'rouge-l':
                RL_f = v1['f']
                RL_f_score.append(RL_f)

# print F-scores

# print(f"R1 Scores: {R1_f_score}")
# print(f"R2 Scores: {R2_f_score}")
# print(f"RL Scores: {RL_f_score}")

# Save results to CSV
df = pd.DataFrame({
    'R1_f': R1_f_score,
    'R2_f': R2_f_score,
    'RL_f': RL_f_score
})


df.to_csv('Result_f_score.csv', index=False)
print("CSV saved successfully.")
