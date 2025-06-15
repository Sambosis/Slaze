from lspo_3d.models.generator import CadQueryGenerator

# Initialize a generator with a standard pre-trained model
# generator = CadQueryGenerator(model_name_or_path="distilgpt2")
generator = CadQueryGenerator(model_name_or_path="codellama/CodeLlama-7b-hf")

# Save the model to a directory
generator.save_model("./stls/lspo_3d/initial_generator")
