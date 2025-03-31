import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://syllabus.sfc.keio.ac.jp"

MAX_PAGE = 108

# セッションを作成
session = requests.Session()

# ヘッダーを設定（User-Agentを偽装）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
}

# ログイン情報
payload = {
    "user[cns_account]": USERNAME,
    "user[password]": PASSWORD,
    "locale": "ja",
    "return_to": "/?locale=ja",
}

# **ログイン**
login_url = "https://syllabus.sfc.keio.ac.jp/users/sign_in"
login_response = session.post(login_url, data=payload, headers=headers)

# **ログイン成功確認**
if "ログアウト" in login_response.text or "ログイン" not in login_response.text:
    print("ログイン成功！")
else:
    print("ログイン失敗")
    exit()


# **すべてのシラバスページのURLを取得**
def get_syllabus_urls():
    syllabus_urls = []
    for page in range(1, MAX_PAGE+1):  # 100ページ分を仮に取得（適宜調整）
        url = f"{BASE_URL}/courses?locale=ja&page={page}&search%5Byear%5D=2024"
        response = session.get(url, headers=headers)  # セッションを使用
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, "html.parser")
        course_links = soup.select("a[href^='/courses/']")
        for link in course_links:
            syllabus_urls.append(BASE_URL + link["href"])
        time.sleep(1)  # サーバー負荷軽減
    return syllabus_urls


# **各シラバスの詳細情報を取得**
def scrape_syllabus(url):
    response = session.get(url, headers=headers)  # セッションを使用
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")

    def get_text(dt_text, container=soup):
        """特定のdt要素の次のdd要素のテキストを取得"""
        dt_element = container.find("dt", string=dt_text)
        if dt_element:
            dd_element = dt_element.find_next_sibling("dd")
            return dd_element.get_text(strip=True) if dd_element else ""
        return ""

    # 授業計画取得（全回分をリストで結合）
    lecture_schedule = []
    for dt in soup.select("dl.lecture-info dt"):
        title = dt.get_text(strip=True)
        dd = dt.find_next_sibling("dd")
        p = dd.find("p")
        details = p.get_text(strip=True) if p else ""
        lecture_schedule.append(f"{title}: {details}")

    data = [
        {
            "科目名": get_text("科目名", div),
            "研究会テーマ": get_text("研究会テーマ"),
            "学部・研究科": get_text("学部・研究科", div),
            "分野": get_text("分野", div),
            # "登録番号": get_text("登録番号"),
            # "科目ソート": get_text("科目ソート"),
            "単位": get_text("単位", div),
            "開講年度・学期": get_text("開講年度・学期", div),
            # "K-Number": get_text("K-Number"),
            # "アスペクト": get_text("アスペクト"),
            "曜日・時限": get_text("曜日・時限"),
            "英語サポート": get_text("授業で英語サポート")[9:11],
            # "授業教員名": get_text("授業教員名"),
            "実施形態": get_text("実施形態"),
            "授業で使う言語": get_text("授業で使う言語"),
            "履修制限": get_text("履修制限")[:10],
            # "履修条件": get_text("履修条件"),
            # "評価方法": get_text("成績評価") or get_text("評価方法"),  # 表記ゆれ対応
            "講義概要": get_text("講義概要"),  # or get_text("授業概要"),  # 表記ゆれ対応
            "GIGA": get_text("GIGAサティフィケート対象"),
            "主題と目標": get_text("主題と目標"),
            # "授業URL": get_text("授業URL"),
            # "開講場所": get_text("開講場所"),
            # "授業形態": get_text("授業形態"),
            # "参考文献": get_text("参考文献"),
            # "連絡先メールアドレス": get_text("連絡先メールアドレス"),
            "授業計画": " | ".join(lecture_schedule),  # 授業計画を1行に結合
            "URL": url,  # 追加したURL情報
        }   for div in soup.find_all("div", class_="subject")
    ]

    return data


# **メイン処理**
print("シラバスURLを取得中...")
syllabus_urls = get_syllabus_urls()

print(f"取得したシラバスURL数: {len(syllabus_urls)}")

syllabus_data = []
for idx, url in enumerate(syllabus_urls):  # 最初の10件のみ取得（適宜調整）
    print(f"{idx+1}/{len(syllabus_urls)}: {url} の情報を取得中...")
    data_list = scrape_syllabus(url)
    for data in data_list:
        syllabus_data.append(data)
    time.sleep(0.2)  # サーバー負荷軽減

# **CSVに保存**
df = pd.DataFrame(syllabus_data)
df.to_csv("sfc_syllabus.csv", index=False, encoding="utf-8-sig")

print("シラバスデータの取得完了！ CSVに保存しました。")
