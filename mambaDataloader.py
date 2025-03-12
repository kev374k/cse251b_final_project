import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

batch_size = 8
# # # LOAD TRAIN # # #
def load_train(tokenize_function):
    df = pd.read_csv('data/train.tsv', sep='\t', header=None)
    df = df.drop(df.columns[2], axis=1)

    newrows = [] # ONLY HANDLES ROWS WITH 1 EMOTION. COMMENTED CODE ADDS NEW TRAINING ROWS FOR MULTIPLE EMOTIONS
    for i in range(0, df.shape[0]): 
        if "," in df[1].iloc[i]:
            # currEmotion = df[1].iloc[i].split(",")
            # currText = df[0].iloc[i]
            # for j in currEmotion:
            #     newrows.append([currText, j])
            
            # currEmotion = df[1].iloc[i].split(",") # ADDED TO ACCOUNT FOR FIRST EMOTION IN A LIST TO SEE IF MORE DATA HELPS
            # newrows.append([df[0].iloc[i], int(currEmotion[0])])
            pass
        else:
            newrows.append([df[0].iloc[i], int(df[1].iloc[i])])

    df = pd.DataFrame(newrows)
            
    train_dataset = Dataset.from_pandas(pd.DataFrame({'text': df[0] + ' [CLS]', 'emotion': df[1]}))
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'emotion'])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader

# # # LOAD VALIDATION # # #
def load_val(tokenize_function):
    df = pd.read_csv('data/dev.tsv', sep='\t', header=None)
    df = df.drop(df.columns[2], axis=1)

    newrows = [] # ONLY HANDLES ROWS WITH 1 EMOTION. COMMENTED CODE ADDS NEW VALIDATION ROWS FOR MULTIPLE EMOTIONS
    for i in range(0, df.shape[0]): # NEED FIND WAY TO HANDLE IF WE DECIDE TO PURSUE. MULTIPLE EMOTIONS COULD CONFUSE THE MODEL IN THIS FORMAT (LIKE THE TESLA CASE PROF MENTIONED)
        if "," in df[1].iloc[i]:
            # currEmotion = df[1].iloc[i].split(",")
            # currText = df[0].iloc[i]
            # for j in currEmotion:
            #     newrows.append([currText, j])
            # currEmotion = df[1].iloc[i].split(",") # ADDED TO ACCOUNT FOR FIRST EMOTION IN A LIST TO SEE IF MORE DATA HELPS
            # newrows.append([df[0].iloc[i], int(currEmotion[0])])
            pass
        else:
            newrows.append([df[0].iloc[i], int(df[1].iloc[i])])

    df = pd.DataFrame(newrows)
    val_dataset = Dataset.from_pandas(pd.DataFrame({'text': df[0]+ ' [CLS]', 'emotion': df[1]}))
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'emotion'])
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    return val_dataloader

# # # LOAD TEST # # #
def load_test(tokenize_function):
    df = pd.read_csv('data/test.tsv', sep='\t', header=None)
    df = df.drop(df.columns[2], axis=1)

    newrows = [] # ONLY HANDLES ROWS WITH 1 EMOTION. COMMENTED CODE ADDS NEW VALIDATION ROWS FOR MULTIPLE EMOTIONS
    for i in range(0, df.shape[0]): # NEED FIND WAY TO HANDLE IF WE DECIDE TO PURSUE. MULTIPLE EMOTIONS COULD CONFUSE THE MODEL IN THIS FORMAT (LIKE THE TESLA CASE PROF MENTIONED)
        if "," in df[1].iloc[i]:
            # currEmotion = df[1].iloc[i].split(",")
            # currText = df[0].iloc[i]
            # for j in currEmotion:
            #     newrows.append([currText, j])
            # currEmotion = df[1].iloc[i].split(",") # ADDED TO ACCOUNT FOR FIRST EMOTION IN A LIST TO SEE IF MORE DATA HELPS
            # newrows.append([df[0].iloc[i], int(currEmotion[0])])
            pass
        else:
            newrows.append([df[0].iloc[i], int(df[1].iloc[i])])

    df = pd.DataFrame(newrows)
    test_dataset = Dataset.from_pandas(pd.DataFrame({'text': df[0]+ ' [CLS]', 'emotion': df[1]}))
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'emotion'])
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return test_dataloader