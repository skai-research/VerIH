import os 
import pandas as pd


if __name__ == "__main__":
    parquest_folders = ["dataset/qwen3-logic-v0", "dataset/qwen3-jailbreak-v1+"]
    output_folder = "dataset/qwen3-logic-v0-jailbreak-v1+"
    os.makedirs(output_folder, exist_ok=True)
    for i in ["train", "test"]:
        dfs = []
        for j in parquest_folders:
            dfs.append(pd.read_parquet(os.path.join(j, f'{i}.parquet')))

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        merged_df.to_parquet(os.path.join(output_folder, f'{i}.parquet'))
