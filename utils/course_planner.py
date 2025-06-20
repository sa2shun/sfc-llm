"""
Intelligent course planning and recommendation system for SFC students.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import itertools

logger = logging.getLogger(__name__)

class Semester(Enum):
    """学期の定義"""
    SPRING = "春学期"
    FALL = "秋学期"
    INTENSIVE = "集中講義"
    YEAR_LONG = "通年"

class Grade(Enum):
    """学年の定義"""
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    GRADUATE = 5

@dataclass
class CourseInfo:
    """科目情報クラス"""
    subject_name: str
    faculty: bool  # True=学部, False=大学院
    category: str
    credits: int
    semester: str
    year: int
    language: str
    instructor: str
    schedule: str
    summary: str
    prerequisites: List[str] = None
    corequisites: List[str] = None
    difficulty_level: int = 3  # 1-5 scale
    workload_hours: int = 45  # 時間/週
    tags: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.corequisites is None:
            self.corequisites = []
        if self.tags is None:
            self.tags = []

@dataclass
class StudentProfile:
    """学生プロファイル"""
    student_id: str
    grade: Grade
    major: str  # 専攻
    completed_courses: List[str]  # 履修済み科目
    current_courses: List[str]   # 履修中科目
    target_graduation_year: int
    interests: List[str]         # 興味分野
    career_goals: List[str]      # 進路目標
    language_preference: str = "japanese"
    max_credits_per_semester: int = 22
    preferred_difficulty: int = 3  # 1-5 scale
    
class CoursePlanner:
    """インテリジェント履修プランナー"""
    
    def __init__(self):
        self.courses_db: Dict[str, CourseInfo] = {}
        self.course_relationships: Dict[str, List[str]] = {}  # 科目間関係
        self.graduation_requirements = self._load_graduation_requirements()
    
    def _load_graduation_requirements(self) -> Dict[str, Any]:
        """卒業要件の定義"""
        return {
            "総単位数": {
                "学部": 124,
                "大学院": 30
            },
            "必修単位": {
                "基盤科目": 20,
                "専門科目": 60,
                "研究科目": 20,
                "自由科目": 24
            },
            "言語要件": {
                "英語": 8,
                "第二外国語": 4
            },
            "実習要件": {
                "研究プロジェクト": 4,
                "インターンシップ": 2
            }
        }
    
    def add_course(self, course: CourseInfo):
        """科目をデータベースに追加"""
        self.courses_db[course.subject_name] = course
        logger.debug(f"Added course: {course.subject_name}")
    
    def analyze_student_progress(self, profile: StudentProfile) -> Dict[str, Any]:
        """学生の履修進捗を分析"""
        completed_credits = sum(
            self.courses_db[course].credits 
            for course in profile.completed_courses 
            if course in self.courses_db
        )
        
        current_credits = sum(
            self.courses_db[course].credits 
            for course in profile.current_courses 
            if course in self.courses_db
        )
        
        remaining_semesters = max(1, (profile.target_graduation_year - datetime.now().year) * 2)
        
        analysis = {
            "completed_credits": completed_credits,
            "current_credits": current_credits,
            "total_credits": completed_credits + current_credits,
            "remaining_credits": self.graduation_requirements["総単位数"]["学部" if profile.grade.value <= 4 else "大学院"] - completed_credits,
            "remaining_semesters": remaining_semesters,
            "credits_per_semester_needed": 0,
            "on_track": True,
            "warnings": []
        }
        
        if analysis["remaining_credits"] > 0:
            analysis["credits_per_semester_needed"] = analysis["remaining_credits"] / remaining_semesters
            
            if analysis["credits_per_semester_needed"] > profile.max_credits_per_semester:
                analysis["on_track"] = False
                analysis["warnings"].append("履修単位数が上限を超える可能性があります")
        
        return analysis
    
    def recommend_courses(self, 
                         profile: StudentProfile, 
                         semester: Semester, 
                         year: int,
                         max_recommendations: int = 10) -> List[Dict[str, Any]]:
        """科目推薦システム"""
        
        # 履修可能な科目をフィルタリング
        available_courses = []
        
        for course_name, course in self.courses_db.items():
            if self._is_course_eligible(course, profile, semester, year):
                score = self._calculate_recommendation_score(course, profile)
                available_courses.append({
                    "course": course,
                    "score": score,
                    "reasons": self._get_recommendation_reasons(course, profile)
                })
        
        # スコア順にソート
        available_courses.sort(key=lambda x: x["score"], reverse=True)
        
        return available_courses[:max_recommendations]
    
    def _is_course_eligible(self, 
                           course: CourseInfo, 
                           profile: StudentProfile, 
                           semester: Semester, 
                           year: int) -> bool:
        """科目が履修可能かチェック"""
        
        # 既に履修済みまたは履修中
        if course.subject_name in profile.completed_courses + profile.current_courses:
            return False
        
        # 学期が合わない
        if semester.value not in course.semester and course.semester != Semester.YEAR_LONG.value:
            return False
        
        # 開講年度が合わない
        if course.year != year:
            return False
        
        # 学部/大学院の区分
        is_undergrad = profile.grade.value <= 4
        if course.faculty != is_undergrad:
            return False
        
        # 前提科目の確認
        for prereq in course.prerequisites:
            if prereq not in profile.completed_courses:
                return False
        
        return True
    
    def _calculate_recommendation_score(self, 
                                      course: CourseInfo, 
                                      profile: StudentProfile) -> float:
        """推薦スコアを計算"""
        score = 0.0
        
        # 興味分野との一致度
        interest_match = sum(
            1 for interest in profile.interests 
            if interest.lower() in course.summary.lower() or 
               interest.lower() in course.category.lower()
        )
        score += interest_match * 0.3
        
        # 進路目標との関連性
        career_match = sum(
            1 for goal in profile.career_goals
            if goal.lower() in course.summary.lower()
        )
        score += career_match * 0.25
        
        # 難易度の適合性
        difficulty_diff = abs(course.difficulty_level - profile.preferred_difficulty)
        score += (5 - difficulty_diff) * 0.1
        
        # 言語設定との一致
        if profile.language_preference == "english" and "english" in course.language.lower():
            score += 0.2
        elif profile.language_preference == "japanese" and "日本語" in course.language:
            score += 0.15
        
        # カテゴリの多様性ボーナス
        completed_categories = set(
            self.courses_db[course_name].category 
            for course_name in profile.completed_courses 
            if course_name in self.courses_db
        )
        
        if course.category not in completed_categories:
            score += 0.1
        
        return score
    
    def _get_recommendation_reasons(self, 
                                  course: CourseInfo, 
                                  profile: StudentProfile) -> List[str]:
        """推薦理由を生成"""
        reasons = []
        
        # 興味分野
        matching_interests = [
            interest for interest in profile.interests
            if interest.lower() in course.summary.lower() or 
               interest.lower() in course.category.lower()
        ]
        if matching_interests:
            reasons.append(f"あなたの興味分野「{', '.join(matching_interests)}」と関連しています")
        
        # 進路目標
        matching_goals = [
            goal for goal in profile.career_goals
            if goal.lower() in course.summary.lower()
        ]
        if matching_goals:
            reasons.append(f"進路目標「{', '.join(matching_goals)}」に役立ちます")
        
        # 難易度
        if abs(course.difficulty_level - profile.preferred_difficulty) <= 1:
            reasons.append("適切な難易度レベルです")
        
        # 新分野
        completed_categories = set(
            self.courses_db[course_name].category 
            for course_name in profile.completed_courses 
            if course_name in self.courses_db
        )
        
        if course.category not in completed_categories:
            reasons.append("新しい分野を学ぶ機会です")
        
        return reasons
    
    def create_semester_plan(self, 
                           profile: StudentProfile, 
                           semester: Semester, 
                           year: int,
                           target_credits: int = 20) -> Dict[str, Any]:
        """学期の履修計画を作成"""
        
        recommendations = self.recommend_courses(profile, semester, year, 20)
        
        # 単位数を考慮して科目を選択
        selected_courses = []
        total_credits = 0
        
        for rec in recommendations:
            course = rec["course"]
            if total_credits + course.credits <= target_credits:
                selected_courses.append(rec)
                total_credits += course.credits
                
                if total_credits >= target_credits * 0.9:  # 90%達成で十分
                    break
        
        # 時間割の競合チェック
        schedule_conflicts = self._check_schedule_conflicts(selected_courses)
        
        plan = {
            "semester": semester.value,
            "year": year,
            "courses": selected_courses,
            "total_credits": total_credits,
            "schedule_conflicts": schedule_conflicts,
            "balance_analysis": self._analyze_course_balance(selected_courses),
            "workload_estimate": sum(course["course"].workload_hours for course in selected_courses)
        }
        
        return plan
    
    def _check_schedule_conflicts(self, courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """時間割の競合をチェック"""
        conflicts = []
        
        # 簡単な時間割解析（実際にはより詳細な実装が必要）
        for i, course1 in enumerate(courses):
            for j, course2 in enumerate(courses[i+1:], i+1):
                if self._has_time_conflict(course1["course"].schedule, course2["course"].schedule):
                    conflicts.append({
                        "course1": course1["course"].subject_name,
                        "course2": course2["course"].subject_name,
                        "type": "time_conflict"
                    })
        
        return conflicts
    
    def _has_time_conflict(self, schedule1: str, schedule2: str) -> bool:
        """簡単な時間割競合チェック"""
        # 実装を簡略化（実際にはより詳細な時間解析が必要）
        if not schedule1 or not schedule2:
            return False
        
        # 曜日の抽出（月、火、水、木、金）
        days1 = set(day for day in ["月", "火", "水", "木", "金"] if day in schedule1)
        days2 = set(day for day in ["月", "火", "水", "木", "金"] if day in schedule2)
        
        return len(days1.intersection(days2)) > 0
    
    def _analyze_course_balance(self, courses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """科目のバランス分析"""
        categories = {}
        languages = {}
        difficulty_levels = []
        
        for course_rec in courses:
            course = course_rec["course"]
            
            # カテゴリ分析
            categories[course.category] = categories.get(course.category, 0) + 1
            
            # 言語分析
            languages[course.language] = languages.get(course.language, 0) + 1
            
            # 難易度分析
            difficulty_levels.append(course.difficulty_level)
        
        return {
            "category_distribution": categories,
            "language_distribution": languages,
            "average_difficulty": sum(difficulty_levels) / len(difficulty_levels) if difficulty_levels else 0,
            "difficulty_range": max(difficulty_levels) - min(difficulty_levels) if difficulty_levels else 0
        }
    
    def generate_graduation_roadmap(self, profile: StudentProfile) -> Dict[str, Any]:
        """卒業までのロードマップを生成"""
        
        current_year = datetime.now().year
        semesters_until_graduation = []
        
        # 卒業までの学期をリストアップ
        for year in range(current_year, profile.target_graduation_year + 1):
            if year == current_year:
                # 現在の年は残りの学期のみ
                if datetime.now().month <= 6:  # 春学期中
                    semesters_until_graduation.append((Semester.FALL, year))
            else:
                semesters_until_graduation.append((Semester.SPRING, year))
                if year < profile.target_graduation_year:
                    semesters_until_graduation.append((Semester.FALL, year))
        
        roadmap = {
            "student_profile": asdict(profile),
            "graduation_requirements": self.graduation_requirements,
            "semester_plans": [],
            "total_plan": {
                "estimated_credits": 0,
                "requirements_fulfillment": {},
                "recommendations": []
            }
        }
        
        # 各学期の計画を作成
        for semester, year in semesters_until_graduation:
            plan = self.create_semester_plan(profile, semester, year)
            roadmap["semester_plans"].append(plan)
            roadmap["total_plan"]["estimated_credits"] += plan["total_credits"]
            
            # プロファイルを更新（仮想的に履修済みとして追加）
            profile.completed_courses.extend([
                course["course"].subject_name for course in plan["courses"]
            ])
        
        return roadmap

# グローバルインスタンス
_course_planner: Optional[CoursePlanner] = None

def get_course_planner() -> CoursePlanner:
    """履修プランナーのインスタンスを取得"""
    global _course_planner
    
    if _course_planner is None:
        _course_planner = CoursePlanner()
    
    return _course_planner

def create_student_profile_from_query(query: str, 
                                    existing_profile: Optional[StudentProfile] = None) -> StudentProfile:
    """
    ユーザーのクエリから学生プロファイルを推定/更新
    
    Args:
        query: ユーザーの質問や要求
        existing_profile: 既存のプロファイル（あれば）
        
    Returns:
        更新された学生プロファイル
    """
    
    # デフォルトプロファイル
    if existing_profile is None:
        profile = StudentProfile(
            student_id="guest",
            grade=Grade.SECOND,
            major="総合政策",
            completed_courses=[],
            current_courses=[],
            target_graduation_year=datetime.now().year + 2,
            interests=[],
            career_goals=[]
        )
    else:
        profile = existing_profile
    
    # クエリから興味分野を抽出
    interest_keywords = {
        "プログラミング": ["プログラミング", "プログラム", "coding", "python", "java"],
        "データサイエンス": ["データサイエンス", "機械学習", "AI", "統計", "分析"],
        "経済": ["経済", "ファイナンス", "金融", "economics", "finance"],
        "政策": ["政策", "public policy", "行政", "government"],
        "メディア": ["メディア", "デザイン", "映像", "グラフィック"],
        "環境": ["環境", "サステナビリティ", "持続可能", "green"],
        "国際関係": ["国際", "グローバル", "international", "外交"]
    }
    
    query_lower = query.lower()
    
    for interest, keywords in interest_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            if interest not in profile.interests:
                profile.interests.append(interest)
    
    # 進路目標の推定
    career_keywords = {
        "エンジニア": ["エンジニア", "engineer", "開発", "プログラマー"],
        "コンサルタント": ["コンサル", "consulting", "戦略"],
        "研究者": ["研究", "research", "大学院", "アカデミック"],
        "起業": ["起業", "startup", "entrepreneur", "ビジネス"],
        "公務員": ["公務員", "行政", "政府", "官僚"]
    }
    
    for career, keywords in career_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            if career not in profile.career_goals:
                profile.career_goals.append(career)
    
    return profile