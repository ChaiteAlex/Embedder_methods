
import pandas as pd

items= [".\\FSTEC_merge_mapping\\final\\object_params.parquet",
        ".\\FSTEC_merge_mapping\\final\\objects.parquet",
        ".\\FSTEC_merge_mapping\\final\\texts.parquet",
        ".\\FSTEC_merge_mapping\\final\\params.parquet",
                ]

for filepath in items:
    df = pd.read_parquet(filepath)
    with open(file=f"{filepath[filepath.rfind("\\")+1:filepath.rfind(".")]}.txt",mode="w", encoding="utf-8") as txt:
            print(df.to_string(index=False))