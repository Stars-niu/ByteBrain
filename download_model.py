from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IEITYuan/Yuan2-2B-Mars-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(".")
tokenizer.save_pretrained(".")