# 🧠 Openfoodfacts Image to text + Finetuning with Unsloth + vLLM Deployment

This project demonstrates how to preprocess, finetune and deploy a 
Vision-Language model. 

The model learns to extract nutrition information from food packaging images.

---

Note : All the flows are shown in /images folder


---

## Data 
- https://huggingface.co/datasets/openfoodfacts/product-database
- Download the csv file and place it in datafolder and run the Notebook in notebook folder to get train, validation and test datasets

## 🍽️ Dataset Description
- Open Food Facts is a database of food products with ingredients, allergens, nutrition facts 

## 🍽️ Datacleaning and EDA
- Please review Notebooks/EDA_dataclean.ipynb for creating the training, validation and test files that can be used
training, validation and testing.

## 🍽️ Nutrition Parameters (per 100g)

This table outlines common nutrition-related fields typically found on food packaging. All values are standardized **per 100 grams** of the product.

| **Parameter**            | **Description**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| `energy_100g`            | Energy content, usually in kilojoules (kJ) or kilocalories (kcal), per 100g     |
| `fat_100g`               | Total fat content per 100g of product                                           |
| `saturated-fat_100g`     | Amount of saturated fat per 100g                                                |
| `trans-fat_100g`         | Amount of trans fat per 100g (often limited or discouraged in diets)            |
| `cholesterol_100g`       | Cholesterol amount per 100g (more common in US labeling than EU)                |
| `carbohydrates_100g`     | Total carbohydrate content per 100g                                             |
| `sugars_100g`            | Sugars (a subset of carbs) per 100g                                             |
| `fiber_100g`             | Dietary fiber per 100g                                                          |
| `proteins_100g`          | Protein content per 100g                                                        |
| `salt_100g`              | Salt content per 100g (used mainly in Europe; converted from sodium)            |
| `sodium_100g`            | Sodium content per 100g (used mainly in the US and scientific contexts)         |

##  🍽️ Training Pipeline (per 100g)
- python train.py

## 🧠 Inference (with Gradio UI)
- python inference.py

## 🔁 Merge LoRA Model for Deployment (optional)
- from peft import AutoPeftModelForCausalLM
- model = AutoPeftModelForCausalLM.from_pretrained("lora_model", device_map="cpu")
- model.save_pretrained("merged_model")

## 🌐 Deploy with vLLM
- chmod +x run_vllm_server.sh
- ./run_vllm_server.sh

## 🧪 Example Python Request to vLLM
- Refer request.py
 
## 🧪 1. Text Quality KPIs
### ✅ BLEU score - Measures Ngram overlap
### ✅ ROUGE - Longest common subsequence
### ✅ RMSE / R2 Score of nutrition values. Extracted from json.

## References
- https://www.kaggle.com/code/alexandrelemercier/cleaning-the-open-food-facts-database
- https://www.kaggle.com/code/michaelfumery/openfoodfacts-data-cleaning
