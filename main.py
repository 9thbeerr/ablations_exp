from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
