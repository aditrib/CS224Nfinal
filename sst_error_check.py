import pandas as pd

input_file = 'data/ids-sst-dev'
output_file = 'yinyang-steadyhand/sst-dev-output.csv'

df_input = pd.read_csv(input_file, delimiter='\t', header=None, names=['id', 'sentence', 'sentiment'])
df_input['id'] = df_input['id'].str.strip()

df_output = pd.read_csv(output_file, delimiter=',', skiprows=1, header=None, names=['id', 'Predicted_Sentiment'])
df_output['id'] = df_output['id'].str.strip()

df_merged = pd.merge(df_input, df_output, on='id')

df_merged['sentiment'] = df_merged['sentiment'].astype(int)
df_merged['Predicted_Sentiment'] = df_merged['Predicted_Sentiment'].astype(int)

mismatches = df_merged[df_merged['sentiment'] != df_merged['Predicted_Sentiment']]

false_negatives = mismatches[mismatches['sentiment'] > mismatches['Predicted_Sentiment']]
false_positives = mismatches[mismatches['sentiment'] < mismatches['Predicted_Sentiment']]

combined_samples = pd.concat([false_positives, false_negatives])

combined_samples.to_csv('false_sst.csv', index=False)
