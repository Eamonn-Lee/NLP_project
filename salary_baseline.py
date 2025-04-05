import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
import math

# Load CSV
file_path = '/Users/garry/PycharmProjects/6713/test/job_data_files/salary_labelled_development_set.csv'
test_file_path = '/Users/garry/PycharmProjects/6713/test/job_data_files/salary_labelled_test_set.csv'
ignore_id_path = '/Users/garry/PycharmProjects/6713/test/err_salary_develpment.csv'


df = pd.read_csv(file_path)
df_ignore = pd.read_csv(ignore_id_path)

# some y_true is impossible
df = df[~df['job_id'].isin(df_ignore['job_id'])]
# Country to currency mapping
country_currency_map = {
    "PH": "PHP", "AUS": "AUD", "NZ": "NZD", "SG": "SGD",
    "MY": "MYR", "TH": "THB", "ID": "IDR", "HK": "HKD"
}

exchange_rates_to_aud = {
    "PHP": 0.029,  # Philippine Peso
    "NZD": 0.92,  # New Zealand Dollar
    "SGD": 1.22,  # Singapore Dollar
    "MYR": 0.37,  # Malaysian Ringgit
    "THB": 0.048,  # Thai Baht
    "IDR": 0.0001,  # Indonesian Rupiah
    "HKD": 0.21,  # Hong Kong Dollar
    "AUD": 1.00  # Australian Dollar (base)
}
normalize_unit = {
    "PHP": 13.98,
    "NZD": 1.21,
    "SGD": 2.18,
    "MYR": 11.57,
    "THB": 6.96,
    "IDR": 19.43,
    "HKD": 2.04,
    "AUD": 1.00
}


def clean_html_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')

    # <tag> -> " "
    for tag in soup.find_all(True):
        tag.insert_before(' ')
        tag.unwrap()

    return soup.get_text()


def detect_unit_extended(window_text, avg, currency):
    """
    Detect the unit information within the given text window.
    Supports five cases: hourly, daily, weekly, monthly, and annual.
    The function searches for all possible matches using regular expressions
    and returns the unit corresponding to the earliest match.
    If no match is found, it defaults to "monthly".
    """
    unit_patterns = {
        "HOURLY": r'(per\s+hour|hourly|時薪|每小時|每小時薪資)',
        "DAILY": r'(per\s+day|daily|日薪|每天|每日薪資)',
        "WEEKLY": r'(per\s+week|weekly|週薪|每週|周薪|每周薪資)',
        "MONTHLY": r'(per\s+month|monthly|月薪|每月|每月薪資)',
        "ANNUAL": r'(per\s+year|yearly|annually|remuneration|super|年薪|每年|每年薪資|年度薪資)'
    }
    matches = []
    for unit, pattern in unit_patterns.items():
        m = re.search(pattern, window_text, re.IGNORECASE)
        if m:
            matches.append((m.start(), unit))
    if matches:
        # print(matches)
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
    else:
        return infer_period_by_amount(avg, currency)


# Infer period based on amount
def infer_period_by_amount(amount, currency):
    amount = int(amount)

    rate_to_aud = exchange_rates_to_aud.get(currency)
    alpha = normalize_unit.get(currency)
    unit = amount * rate_to_aud * alpha

    if currency == 'AUD' or currency == "NZD":
        if unit < 3000:
            return "HOURLY"
        else:
            return "ANNUAL"
    else:
        if unit < 40:
            return "HOURLY"
        elif unit < 400:
            return "DAILY"
        elif unit < 3000:
            return "WEEKLY"
        else:
            return "MONTHLY"


def get_fixed_from_posi(posi_result):
    # posi_result = sorted(posi_result,
    #                      key=lambda x: (int(re.findall(r'\d+', x[1])[0]) + int(re.findall(r'\d+', x[1])[1])) / 2)
    has_true = any(flag for flag, _ in posi_result)
    if has_true:
        range_text = next(val for flag, val in posi_result if flag)
        min_salary, max_salary, _ = range_text.split('-')[:3]

        min_salary = round(float(min_salary))
        # min_salary = math.floor(float(min_salary))
        max_salary = round(float(max_salary))

        for (is_range, text) in posi_result:
            if not is_range:
                fix_salary, _ = text.split('-')[:2]
                fix_salary = round(float(fix_salary))
                if min_salary < fix_salary < max_salary:
                    return text
        else:
            return range_text
    else:
        return posi_result[0][1]


def extract_salary_with_inference(text, nation_code):
    text = text.replace(",", "")
    for _, value in country_currency_map.items():
        text = text.replace(value, "$")
    text = text.replace("RM", "$")
    text = text.replace("฿", "$")
    text = text.replace("AU", "$")

    text = text.replace("$$", "$")

    text = text.replace("  ", " ")
    text = re.sub(r'([\u4e00-\u9fff])', r'\1 ', text)

    text = clean_html_tags(text)
    tokens = text.split()
    currency = country_currency_map.get(nation_code, "None")
    posi_result = []

    # Iterate over tokens to find salary-related keywords
    for i, token in enumerate(tokens):
        token = token.lower().strip(":")
        if (token in ['待遇', 'salary', 'wage', 'compensation', 'remuneration', 'gaji', 'bermula', 'basic', 'pokok',
                      'income']
                or "薪" in token or "$" in token or "¥" in token or "₱" in token ):
            # print(token)
            # Define a window (using the next 6 tokens as context)
            end = min(i + 6, len(tokens))
            window = tokens[i:end]
            window_text = " ".join(window)
            # window_text = re.sub(r'(\d+)k', lambda m: str(int(m.group(1)) * 1000), window_text)

            window_text = re.sub(r'\b[Tt][Oo]\b', '-', window_text)
            window_text = window_text.replace("and", "-")
            window_text = window_text.replace("至", "-")
            window_text = window_text.replace("hingga ke", "-")
            window_text = window_text.replace("hingga", "-")
            window_text = window_text.replace("Hingga", "-")
            window_text = window_text.replace("HINGGA", "-")

            is_range = False

            # Match salary
            pattern = r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b\s*(?:[-–—|TO|to|To])?\s*(?:[€$₱¥₹]|[a-z]{0,3})?\s*\b(\d+(?:,\d+)*(?:\.\d+)?)?\b'
            # F1 > 0.7
            pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:[-–—])?\s*(?:[€$₱¥₹]|[a-z]{0,3})?\s*(\d+(?:,\d+)*(?:\.\d+)?)?'
            match = re.search(pattern, window_text)
            if match:
                min_salary = match.group(1)
                if match.group(2):
                    is_range = True
                max_salary = match.group(2) if match.group(2) else match.group(1)

                min_salary = round(float(min_salary))
                # min_salary = math.floor(float(min_salary))
                max_salary = round(float(max_salary))

                if min_salary > max_salary:
                    continue
                if 2.5 * min_salary <= max_salary:
                    continue
                avg = (int(min_salary) + int(max_salary)) / 2
                # period = detect_unit_extended(window_text, avg, currency)
                period = infer_period_by_amount(avg, currency)
                posi_result.append((is_range, f"{min_salary}-{max_salary}-{currency}-{period}"))

    if posi_result:
        if len(posi_result) > 1:
            result = get_fixed_from_posi(posi_result)
            return result
        else:
            return posi_result[0][1]

    else:
        return "0-0-None-None"


# Apply extractor
df['predicted_salary'] = df.apply(
    lambda row: extract_salary_with_inference(
        f"{row['job_title']} {row['job_ad_details']}",
        row['nation_short_desc']
    ),
    axis=1
)

# TP, FP, TN, FN
TP = np.sum((df['predicted_salary'] == df['y_true']) & (df['y_true'] != "0-0-None-None"))
FP = np.sum((df['predicted_salary'] != df['y_true']) & (df['predicted_salary'] != "0-0-None-None"))
FN = np.sum((df['predicted_salary'] == "0-0-None-None") & (df['y_true'] != "0-0-None-None"))
TN = np.sum((df['predicted_salary'] == "0-0-None-None") & (df['y_true'] == "0-0-None-None"))

precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
accuracy = (TP + TN) / (FP + FN + TP + TN)

# Print prediction vs ground truth
# print("\n🔍 Prediction vs Ground Truth:\n")
# for i, row in df.iterrows():
#     predicted = row['predicted_salary']
#     expected = row['y_true']
#     if predicted != expected:
#         print(f"[{i}] ❌ Predicted: {predicted} | Expected: {expected}")
#         print(f"{row['job_id']} {row['job_title']} {row['job_ad_details']}")
#         print()
    # else:
    #     print(f"[{i}] ✅ Matched:   {predicted}")

print("Development dataset:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print()

# test set
df = pd.read_csv(test_file_path)

# Apply extractor
df['predicted_salary'] = df.apply(
    lambda row: extract_salary_with_inference(
        f"{row['job_title']} {row['job_ad_details']}",
        row['nation_short_desc']
    ),
    axis=1
)


TP = np.sum((df['predicted_salary'] == df['y_true']) & (df['y_true'] != "0-0-None-None"))
FP = np.sum((df['predicted_salary'] != df['y_true']) & (df['predicted_salary'] != "0-0-None-None"))
FN = np.sum((df['predicted_salary'] == "0-0-None-None") & (df['y_true'] != "0-0-None-None"))
TN = np.sum((df['predicted_salary'] == "0-0-None-None") & (df['y_true'] == "0-0-None-None"))


precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
accuracy = (TP + TN) / (FP + FN + TP + TN)

print("Test dataset:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

'''
Development dataset:
Precision: 0.7811
Recall: 0.9370
F1 Score: 0.8519
Accuracy: 0.8489

Test dataset:
Precision: 0.7412
Recall: 0.9206
F1 Score: 0.8212
Accuracy: 0.8219
'''
