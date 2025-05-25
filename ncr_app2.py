import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openai
import os
import io
import warnings

st.set_page_config(layout="wide")

page = st.sidebar.radio("Navigate", [
    "NCR Raw Data",
    "Similarity Tool",
    "LLM Review",
    "Cost Attribution",
    "Ops Summary Chart",
    "NCR Timeline"
])

@st.cache_data
def load_data(file):
    ncr_df = pd.read_excel(file, sheet_name="NCR Data")
    cost_df = pd.read_excel(file, sheet_name="Costings")
    ops_df = pd.read_excel(file, sheet_name="Operations Summary")
    return ncr_df, cost_df, ops_df

uploaded_file = st.file_uploader("Upload NCR Excel file", type="xlsx")

warnings.filterwarnings("ignore", category=FutureWarning)

if 'llm_summaries' not in st.session_state:
    st.session_state['llm_summaries'] = {}

if uploaded_file:
    ncr_df, cost_df, ops_df = load_data(uploaded_file)

    if page == "NCR Raw Data":
        st.title("NCR Raw Data")
        col = st.selectbox("Filter column:", ncr_df.columns.tolist())
        search = st.text_input("Search")
        filtered = ncr_df[ncr_df[col].astype(str).str.contains(search, case=False)] if search else ncr_df
        st.dataframe(filtered)

    elif page == "Similarity Tool":
        st.title("NCR Similarity Tool")
        selected_ncr = st.selectbox("Select NCR", ncr_df['NCR Number'])
        selected_row = ncr_df[ncr_df['NCR Number'] == selected_ncr].iloc[0]

        def extract_keywords(text):
            return set(re.findall(r"\b(drill|sealant|hole|rivet|tool|drawing|spec)\b", text.lower()))

        def get_similarity_score(ref_row, comp_row):
            score = 0
            text_sim = SequenceMatcher(None, ref_row['Discrepancy Text'], comp_row['Discrepancy Text']).ratio()
            score += text_sim * 0.4
            if ref_row['Part Number'] == comp_row['Part Number']:
                score += 0.2
            if any(tool in comp_row['Discrepancy Text'] for tool in re.findall(r"TL-\d+", ref_row['Discrepancy Text'])):
                score += 0.1
            if any(dwg in comp_row['Discrepancy Text'] for dwg in re.findall(r"DWG-\d+|SPEC-\d+", ref_row['Discrepancy Text'])):
                score += 0.1
            score += len(extract_keywords(ref_row['Discrepancy Text']).intersection(extract_keywords(comp_row['Discrepancy Text']))) * 0.05
            return round(min(score, 1.0) * 100, 1)

        if 'threshold' not in st.session_state:
            st.session_state['threshold'] = 80.0
        threshold = st.slider("Similarity Threshold (%)", 0.0, 100.0, st.session_state['threshold'], step=1.0, key='thresh_slider')
        st.session_state['threshold'] = threshold

        scores = [(row['NCR Number'], get_similarity_score(selected_row, row)) for _, row in ncr_df.iterrows() if row['NCR Number'] != selected_ncr]
        result_df = ncr_df.set_index('NCR Number').copy()
        result_df['Similarity %'] = 0
        for ncr_num, sim in scores:
            result_df.at[ncr_num, 'Similarity %'] = sim
        result_df = result_df.reset_index()
        filtered_df = result_df[result_df['Similarity %'] >= threshold].sort_values(by='Similarity %', ascending=False)

        st.subheader(f"Total Matches: {len(filtered_df)}")

        def create_bar(value): return f"{'█' * int(value / 5)} {value:.1f}%"
        filtered_df['Similarity'] = filtered_df['Similarity %'].apply(create_bar)

        st.dataframe(filtered_df[['NCR Number', 'Part Number', 'Discrepancy Text', 'Similarity']])

        # --- LLM PAIRWISE TO SELECTED NCR ONLY ---
        if st.button("Run LLM Analysis on Matches"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            # Build prompt and context for pairwise analysis
            ref_text = selected_row['Discrepancy Text']
            ref_part = selected_row['Part Number']
            system_prompt = f"""
You are an aerospace quality engineer. The reference NCR is:
NCR: {selected_ncr}, Part: {ref_part}
Text: {ref_text}
For each candidate NCR below, state if it describes the SAME TYPE OF ISSUE and would have the SAME RESOLUTION as the reference NCR, regardless of part/tool/drawing, based on the fundamental FAULT. Mark ✓ for same, ✗ for different. Briefly explain why. Output as a markdown table: | NCR Number | Tick/Cross | Reason |
"""
            ncr_list = filtered_df[['NCR Number', 'Part Number', 'Discrepancy Text']].to_dict(orient='records')
            ncr_text = "\n".join([f"{n['NCR Number']} (Part: {n['Part Number']}): {n['Discrepancy Text']}" for n in ncr_list])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ncr_text}
            ]
            with st.spinner("Asking the LLM..."):
                response = openai.chat.completions.create(model="gpt-4", messages=messages)
                explanation = response.choices[0].message.content
                # Robust markdown table extraction
                try:
                    table_start = explanation.index('|')
                    table_text = explanation[table_start:]
                    table_df = pd.read_csv(io.StringIO(table_text), sep='|').dropna(axis=1, how='all')
                    table_df.columns = [col.strip() for col in table_df.columns]
                    table_df = table_df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
                    ticked_df = table_df[table_df["Tick/Cross"] == "✓"]
                    final_table = pd.merge(ticked_df, filtered_df, on="NCR Number", how="left")
                    st.session_state['llm_explanation'] = explanation
                    st.session_state['llm_table'] = final_table[["NCR Number", "Part Number", "Discrepancy Text", "Similarity", "Reason"]]
                    st.session_state['ticked_cost_ncrs'] = final_table['NCR Number'].tolist()
                    # Store for summary in LLM Review
                    st.session_state['llm_discrepancy_texts'] = final_table['Discrepancy Text'].tolist()
                    st.session_state['llm_disposition_texts'] = ncr_df[ncr_df['NCR Number'].isin(final_table['NCR Number'])]['Disposition Text'].tolist()
                    st.session_state['llm_ticked_cost'] = cost_df[cost_df['NCR Number'].isin(final_table['NCR Number'])]['Total Cost (£)'].sum()
                    st.session_state['llm_summary_prompted'] = False
                    st.session_state['llm_summaries'] = {}
                except Exception as e:
                    st.session_state['llm_explanation'] = explanation
                    st.session_state['llm_table'] = pd.DataFrame()
            st.success("LLM analysis complete.")

    elif page == "LLM Review":
        st.title("LLM Check on Selected NCRs")
        if 'llm_table' in st.session_state and not st.session_state['llm_table'].empty:
            st.dataframe(st.session_state['llm_table'])
        else:
            st.info("Run the Similarity Tool and click 'Run LLM Analysis' first.")

        # Only generate summaries if they haven't already been generated for this batch
        if ('llm_discrepancy_texts' in st.session_state and st.session_state['llm_discrepancy_texts'] and
            'llm_disposition_texts' in st.session_state and st.session_state['llm_disposition_texts'] and
            not st.session_state['llm_summaries']):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            discrepancy_texts = "\n".join(st.session_state['llm_discrepancy_texts'])
            disposition_texts = "\n".join(st.session_state['llm_disposition_texts'])
            total_cost = st.session_state['llm_ticked_cost']

            # Prepare analysis prompt
            summary_prompt = f"""
You are an expert aerospace engineering analyst.
Below is a batch of NCR discrepancy texts and disposition texts, with associated cost {total_cost:,.2f} pounds.
First, write a detailed, readable summary for a non-technical manager explaining:
- The nature of these NCRs as a group (patterns, common issues, recurring part numbers, aircraft numbers, operator or tool patterns, etc),
- The total cost, and any patterns that explain the cost,
- What can be learned or actioned, referencing specific parts, aircraft, drawings, operators, tools or specifications if relevant.

Discrepancy Texts:
{discrepancy_texts}

Then, after a divider, write a similarly detailed engineering summary of all the DISPOSITION texts combined, referencing the way the issues were resolved, specifications and tools used, and giving a sense of quality control robustness.

Disposition Texts:
{disposition_texts}

Your output should be in two clearly labelled sections: 
**Combined Discrepancy Summary** (first), and 
**Combined Disposition Summary** (second). 
Use bullet points or paragraphs as you see fit, but be thorough and easy to follow.
"""
            with st.spinner("Summarising batch with LLM..."):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": summary_prompt}],
                    max_tokens=1200
                )
                summary_text = response.choices[0].message.content
                st.session_state['llm_summaries'] = summary_text

        if st.session_state['llm_summaries']:
            # Split the two summaries and display them in separate boxes
            summary_text = st.session_state['llm_summaries']
            disc_label = "**Combined Discrepancy Summary**"
            disp_label = "**Combined Disposition Summary**"
            if disc_label in summary_text and disp_label in summary_text:
                disc_text = summary_text.split(disc_label)[-1].split(disp_label)[0].strip()
                disp_text = summary_text.split(disp_label)[-1].strip()
                st.markdown(f"#### Combined Discrepancy Summary")
                st.info(disc_text)
                st.markdown(f"#### Combined Disposition Summary")
                st.info(disp_text)
            else:
                st.info(summary_text)

    elif page == "Cost Attribution":
        st.title("Cost Attribution")
        if 'ticked_cost_ncrs' in st.session_state:
            merged = pd.merge(ncr_df[ncr_df['NCR Number'].isin(st.session_state['ticked_cost_ncrs'])], cost_df, on='NCR Number', how='left')
        else:
            merged = pd.merge(ncr_df, cost_df, on='NCR Number', how='left')
        merged['Total Cost (£)'] = merged['Total Cost (£)'].fillna(0)
        st.dataframe(merged)
        st.metric("Total Cost", f"£{merged['Total Cost (£)'].sum():,.2f}")

    elif page == "Ops Summary Chart":
        st.title("Top Orders by NCR Rate (%) and Total Cost")
        if ops_df is None or ops_df.empty:
            st.warning("No 'Operations Summary' sheet found in your upload.")
        else:
            st.write("This chart shows the worst performing orders (highest NCR rate) at the top, with NCR rate and cost on separate axes.")
            cost_map = dict(zip(cost_df['NCR Number'], cost_df['Total Cost (£)']))
            ncr_costs = ncr_df[['Order Number', 'NCR Number']].copy()
            ncr_costs['Total Cost (£)'] = ncr_costs['NCR Number'].map(cost_map).fillna(0)
            order_costs = ncr_costs.groupby('Order Number')['Total Cost (£)'].sum().reset_index()
            ops_with_cost = ops_df.merge(order_costs, on='Order Number', how='left').fillna(0)
            sorted_ops = ops_with_cost.sort_values(by='NCR Rate (%)', ascending=False)
            st.dataframe(sorted_ops)
            fig, ax1 = plt.subplots(figsize=(14, 8))
            top_n = min(25, len(sorted_ops))
            orders = sorted_ops['Order Number'][:top_n][::-1]
            ncr_rates = sorted_ops['NCR Rate (%)'][:top_n][::-1]
            costs = sorted_ops['Total Cost (£)'][:top_n][::-1]
            y = range(len(orders))
            bar_height = 0.35
            bars1 = ax1.barh([val + bar_height/2 for val in y], ncr_rates, height=bar_height, color='#C0392B', label='NCR Rate (%)')
            ax1.set_xlabel('NCR Rate (%)', color='#C0392B')
            ax1.set_ylabel('Order Number')
            ax1.tick_params(axis='x', labelcolor='#C0392B')
            ax1.set_yticks(y)
            ax1.set_yticklabels(orders)
            ax2 = ax1.twiny()
            bars2 = ax2.barh([val - bar_height/2 for val in y], costs, height=bar_height, color='#23395D', label='Total Cost (£)')
            ax2.set_xlabel('Total Cost (£)', color='#23395D')
            ax2.tick_params(axis='x', labelcolor='#23395D')
            for i, (rate, cost) in enumerate(zip(ncr_rates, costs)):
                ax1.text(rate + max(ncr_rates)*0.01, i + bar_height/2, f"{rate:.2f}%,", va='center', color='black', fontsize=8, fontweight='bold')
                ax2.text(cost + max(costs)*0.01, i - bar_height/2, f"£{cost:,.0f}", va='center', color='black', fontsize=8, fontweight='bold')
            ax1.set_title("Top Orders by NCR Rate (%) and Total Cost")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
            fig.tight_layout()
            st.pyplot(fig)

    # ---- NCR Timeline (from working app1 version) ----
    elif page == "NCR Timeline":
        st.title("NCR Timeline Viewer")
        st.write("Visualise the timeline of NCRs by Aircraft Number, Part Number, or Order Number.")

        timeline_mode = st.radio("Choose how to view timeline:", ("Aircraft Number", "Part Number", "Order Number"))
        if timeline_mode == "Aircraft Number":
            options = sorted([x for x in ncr_df['Aircraft Number'].dropna().unique() if str(x).strip()])
        elif timeline_mode == "Part Number":
            options = sorted([x for x in ncr_df['Part Number'].dropna().unique() if str(x).strip()])
        else:
            options = sorted([x for x in ncr_df['Order Number'].dropna().unique() if str(x).strip()])

        if not options:
            st.info(f"No valid {timeline_mode} values found in NCR data.")
        else:
            selection = st.selectbox(f"Select {timeline_mode}", options)
            if timeline_mode == "Aircraft Number":
                df_timeline = ncr_df[ncr_df['Aircraft Number'] == selection]
            elif timeline_mode == "Part Number":
                df_timeline = ncr_df[ncr_df['Part Number'] == selection]
            else:
                df_timeline = ncr_df[ncr_df['Order Number'] == selection]

            if not df_timeline.empty:
                df_timeline = df_timeline.copy()
                df_timeline['Created Date'] = pd.to_datetime(df_timeline['Created Date'], errors='coerce')
                df_timeline = df_timeline.sort_values('Created Date')
                st.dataframe(df_timeline)

                status_map = {"Closed": "#1a9b1a", "Cancelled": "#d81b60", "Open": "#1976d2"}
                fig, ax = plt.subplots(figsize=(14, 2.5))
                min_date = df_timeline['Created Date'].min()
                max_date = df_timeline['Created Date'].max()
                ax.axhline(0, color='black', linewidth=3, zorder=1)
                for i, (idx, row) in enumerate(df_timeline.iterrows()):
                    status = str(row['NCR Status']).strip().capitalize()
                    colour = status_map.get(status, "#888888")
                    ax.scatter(row['Created Date'], 0, marker="o", s=120, color=colour, zorder=2, edgecolor='white', linewidth=1.2)
                    y_label = 0.23 if i % 2 == 0 else -0.23
                    ax.text(row['Created Date'], y_label, str(row['NCR Number']), ha='center', fontsize=8, rotation=38, fontweight='bold')
                # Add left/right date text
                if pd.notnull(min_date):
                    ax.text(min_date, -0.35, min_date.strftime('%Y-%m-%d'), ha='center', fontsize=8, color='black', fontweight='bold')
                if pd.notnull(max_date):
                    ax.text(max_date, -0.35, max_date.strftime('%Y-%m-%d'), ha='center', fontsize=8, color='black', fontweight='bold')
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlabel("Date")
                ax.set_ylim(-0.4, 0.35)
                if pd.notnull(min_date) and pd.notnull(max_date):
                    ax.set_xlim(min_date, max_date)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"No NCRs found for selected {timeline_mode}.")

else:
    st.info("Please upload an Excel file with NCR Data, Costings, and Operations Summary sheets.")
