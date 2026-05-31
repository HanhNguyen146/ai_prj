from datasets import load_dataset
ds = load_dataset('taesiri/imagenet-hard', split='validation')
ds.save_to_disk('./data/imagenet-hard')