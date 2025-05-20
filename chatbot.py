# Requisitos: pip install streamlit transformers peft datasets pandas
import os
import json
import time
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

############################################
# Función Principal de la Página
############################################

def createPage():

    # ─── Configuración de rutas ──────────────────────────────────────────────────
    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH      = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
    TIMESTAMP_FILE = os.path.join(BASE_DIR, ".data_timestamp.json")
    MODEL_DIR      = os.path.join(BASE_DIR, "Models", "mistral_rfi_lora")
    # Identificador de modelo base en Hugging Face (requiere login HF CLI)
    BASE_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"

    @st.cache_resource
    def load_dataset(path):
        df = pd.read_excel(path)
        df = df.dropna(subset=["Pregunta", "Respuesta"]).reset_index(drop=True)
        df["text"] = df.apply(lambda x: f"<|prompt|>{x['Pregunta']}<|response|>{x['Respuesta']}", axis=1)
        return Dataset.from_pandas(df[["text"]])

    @st.cache_resource
    def get_model_and_tokenizer():
        # Asegúrate de haber hecho 'huggingface-cli login' con tu token de acceso
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            use_auth_token=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=False,
            torch_dtype="auto",
            device_map="cpu",
            use_auth_token=True
        )
        # Preparar LoRA en CPU
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer

    def data_changed(path, ts_file):
        mtime = os.path.getmtime(path)
        if os.path.exists(ts_file):
            with open(ts_file, 'r') as f:
                info = json.load(f)
            if info.get('mtime') == mtime:
                return False
        with open(ts_file, 'w') as f:
            json.dump({'mtime': mtime}, f)
        return True

    def fine_tune(model, tokenizer, dataset):
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            save_total_limit=1,
            logging_steps=50,
            fp16=False,
            push_to_hub=False,
            hub_strategy="never",
            hub_model_id=None
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

    try:
        st.title("Asistente Virtual DIPRO")
        if data_changed(DATA_PATH, TIMESTAMP_FILE) or not os.path.isdir(MODEL_DIR):
            st.info("Detectando cambios en Data RFI, entrenando modelo...")
            dataset = load_dataset(DATA_PATH)
            model, tokenizer = get_model_and_tokenizer()
            fine_tune(model, tokenizer, dataset)
            st.success("Entrenamiento completado y modelo guardado.")
        else:
            st.success("Modelo entrenado y actualizado.")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR,
                use_fast=True
            )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )

        st.subheader("Chatea con tu Asistente:")
        user_input = st.text_input("Pregunta:")
        if st.button("Enviar") and user_input:
            prompt = f"<|prompt|>{user_input}<|response|>"
            with st.spinner("Generando respuesta..."):
                out = generator(prompt)
            response = out[0]['generated_text'].split('<|response|>')[-1]
            st.write(response)

        st.write(f"Última actualización de datos: {time.ctime(os.path.getmtime(DATA_PATH))}")
    except Exception as e:
        st.error("Error al procesar el archivo 'Data RFI.xlsx'.")
        st.error(str(e))
        return

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    return True
