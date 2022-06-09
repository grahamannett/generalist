# incorporate this into a test for at least this dataset
# parser = argparse.ArgumentParser()
# parser.add_argument("--aokvqa_dir", type=str, default=AOKVQA_DIR)
# parser.add_argument("--split", type=str, default="train")
# parser.add_argument("--version", type=str, default="v1p0")
# args = parser.parse_args()

# aokvqa_dataset = AokvqaDataset(args.aokvqa_dir, args.split, args.version)
# aokvqa_dataset_val = AokvqaDataset(args.aokvqa_dir, "val", args.version)
# # print(len(aokvqa_dataset))

# data1 = aokvqa_dataset[0]
# val1 = aokvqa_dataset_val[0]
# filepath1 = data1.coco_path
# filepath2 = val1.coco_path

# print(os.path.exists(filepath1))
# print(os.path.exists(filepath2))
# breakpoint()
