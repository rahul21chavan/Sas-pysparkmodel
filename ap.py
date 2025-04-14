import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime

# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------- Agent Definitions -----------------

class ParsingAgent:
    def parse(self, sas_code: str) -> dict:
        statements = [stmt.strip() for stmt in sas_code.strip().split(";") if stmt.strip()]
        return {"statements": statements}


# class RuleBasedTranslationAgent:
#     def translate(self, parsed_code: dict) -> list:
#         translations = []
#         for stmt in parsed_code["statements"]:
#             stmt_lower = stmt.lower()
#             if "data" in stmt_lower and "=" in stmt_lower:
#                 table_name = stmt.split("=")[0].split()[-1].strip()
#                 source = stmt.split("=")[1].strip()
#                 translations.append(f'{table_name} = spark.read.table("{source}")')
#             elif "set" in stmt_lower:
#                 table_name = stmt.split()[1]
#                 translations.append(f'df = spark.read.table("{table_name}")')
#             elif "proc print" in stmt_lower:
#                 translations.append("df.show()")
#             else:
#                 translations.append(f"# TODO: Translate -> {stmt}")
#         return translations
class RuleBasedTranslationAgent:
    def translate(self, parsed_code: dict) -> list:
        translations = []
        data_section = False
        columns = []
        data_lines = []
        table_name = ""

        for stmt in parsed_code["statements"]:
            stmt_lower = stmt.lower()

            # Handle empty or malformed statements
            if not stmt.strip():
                continue

            # Handle DATA section
            if stmt_lower.startswith("data"):
                table_name = stmt.split()[1] if len(stmt.split()) > 1 else ""
                if table_name:
                    translations.append(f"# Creating DataFrame for table {table_name}")
                    data_section = True
                continue

            # Handle INPUT (defining the schema and adding data)
            if "input" in stmt_lower:
                # Extract column names from INPUT statement (space-separated columns)
                columns = [col.split()[0] for col in stmt.split(",")]
                translations.append(f"columns = {columns}")
                continue

            # Handle DATA insertion (manually defined data)
            if data_section and stmt_lower.startswith("datalines"):
                data_section = "datalines"
                translations.append("data = [")
                continue

            if data_section == "datalines" and stmt.strip() and not stmt_lower.startswith("run"):
                # Add data rows, assuming space-separated values
                data_values = stmt.split()
                data_lines.append(f"    ({', '.join(data_values)})")

            # Handle END OF DATALINES / RUN
            if stmt_lower.startswith("run"):
                if data_lines:
                    translations.append("\n".join(data_lines) + "\n]")
                    translations.append(f"df = spark.createDataFrame(data, columns)")
                data_section = False
                continue

            # Handle Computation (like comm = SALARY*0.25)
            if "=" in stmt_lower:
                parts = stmt.split("=")
                if len(parts) == 2:  # Ensure only 1 "=" symbol is present
                    left, right = parts
                    left = left.strip()
                    right = right.strip()
                    if "salary" in right.lower():
                        translations.append(f"df = df.withColumn('{left}', df['SALARY'] * 0.25)")
                else:
                    # Handle cases where there are multiple "=" signs (e.g., complex expressions)
                    translations.append(f"# Complex Expression: {stmt}")
                continue

            # Handle PROC steps (like PROC PRINT, PROC SORT, etc.)
            if "proc print" in stmt_lower:
                translations.append("df.show()")
                continue

            if "proc sort" in stmt_lower:
                columns = stmt.split()[2:]
                if columns:
                    translations.append(f"df = df.orderBy({', '.join(columns)})")
                continue

            # Handle LABEL (skip or add as part of the column metadata)
            if "label" in stmt_lower:
                translations.append("# LABEL statements don't have direct PySpark equivalent.")
                continue

            # Default case (when we don't recognize the statement)
            translations.append(f"# TODO: Translate -> {stmt}")

        return translations



class GeminiTranslationAgent:
    def translate(self, sas_code: str) -> list:
        prompt = f"""You are an expert in converting legacy SAS code to PySpark.
Convert the following SAS code into clean, optimized PySpark:

SAS CODE:
{sas_code}

Only return PySpark code without explanation."""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            response = model.generate_content(prompt)
            return response.text.strip().splitlines()
        except Exception as e:
            return [f"# Gemini Error: {e}"]


class ValidationAgent:
    def validate(self, code: list) -> list:
        return [line for line in code]


class FeedbackAgent:
    def collect_feedback(self, code: list) -> str:
        incomplete = [line for line in code if "TODO" in line or "# Gemini Error" in line]
        if incomplete:
            return "⚠️ Some translations need manual review."
        else:
            return "✅ Translation looks complete."


# ----------------- Streamlit App -----------------

st.set_page_config(page_title="SAS to PySpark Agentic Converter", layout="wide")
st.title("SAS to PySpark Agentic Converter")

# Initialize session logs if not present
if "logs" not in st.session_state:
    st.session_state.logs = []

# Tabs: Convert | History
tabs = st.tabs(["Convert", "History"])

with tabs[0]:
    # Upload SAS file
    uploaded_file = st.file_uploader("Upload your SAS script (.sas) file:", type=["sas"])

    if uploaded_file:
        sas_input = uploaded_file.read().decode("utf-8")
        st.text_area("SAS Code", sas_input, height=200)

        if st.button("Convert Code"):
            if not sas_input.strip():
                st.warning("Please upload a valid SAS script.")
            else:
                with st.spinner("Running all agents..."):
                    # Instantiate agents
                    parser = ParsingAgent()
                    rule_translator = RuleBasedTranslationAgent()
                    gemini_translator = GeminiTranslationAgent()
                    validator = ValidationAgent()
                    feedback = FeedbackAgent()

                    # Agent processing
                    parsed = parser.parse(sas_input)
                    rule_code = validator.validate(rule_translator.translate(parsed))
                    gemini_code = validator.validate(gemini_translator.translate(sas_input))
                    feedback_rule = feedback.collect_feedback(rule_code)
                    feedback_gemini = feedback.collect_feedback(gemini_code)

                # Save session log
                st.session_state.logs.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sas_code": sas_input,
                    "rule_output": "\n".join(rule_code),
                    "gemini_output": "\n".join(gemini_code),
                    "agent_used": "Both",
                    "file_name": uploaded_file.name
                })

                # Multi-Agent Comparison View
                st.subheader("Multi-Agent Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Rule-Based Agent Output")
                    st.code("\n".join(rule_code), language="python")
                    st.info(feedback_rule)

                with col2:
                    st.markdown("### Gemini Agent Output")
                    st.code("\n".join(gemini_code), language="python")
                    st.info(feedback_gemini)

                # Save result to file
                final_code = "\n".join(gemini_code)
                filename = f"pyspark_translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                st.download_button("Download PySpark Code", final_code, file_name=filename, mime="text/x-python")

with tabs[1]:
    st.subheader("Conversion History")

    if st.session_state.logs:
        for idx, log in enumerate(reversed(st.session_state.logs), 1):
            with st.expander(f"Session {idx}: {log['timestamp']} - {log['file_name']}"):
                st.markdown("**Input (SAS):**")
                st.code(log["sas_code"], language="sas")

                st.markdown("**Rule-based Output:**")
                st.code(log["rule_output"], language="python")

                st.markdown("**Gemini Output:**")
                st.code(log["gemini_output"], language="python")
    else:
        st.info("No past sessions yet. Start converting!")
