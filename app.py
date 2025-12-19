import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import hashlib
import json
import random

# Настройка страницы
st.set_page_config(
    page_title="💪 Фитнес Трекер",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 0.5rem;
    }
    .user-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .achievement-card {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .sport-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .training-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9fff9;
    }
</style>
""", unsafe_allow_html=True)

class FitnessApp:
    def __init__(self):
        self.data_dir = 'user_data'
        self._ensure_data_directory()
        # Инициализация базы знаний о тренировках
        self.init_training_knowledge_base()
    
    def _ensure_data_directory(self):
        """Создает папку для данных если её нет"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def init_training_knowledge_base(self):
        """Инициализация базы знаний о тренировках для разных видов спорта"""
        self.sport_types = {
            'Силовые тренировки': {
                'icon': '🏋️',
                'exercises': ['Жим лежа', 'Приседания', 'Становая тяга', 'Жим стоя', 'Тяга штанги'],
                'goals': ['Увеличение силы', 'Набор мышечной массы', 'Улучшение выносливости']
            },
            'Бег/Кардио': {
                'icon': '🏃',
                'exercises': ['Бег', 'Велотренажер', 'Скакалка', 'Плавание', 'Ходьба'],
                'goals': ['Снижение веса', 'Улучшение выносливости', 'Подготовка к забегу']
            },
            'Йога/Пилатес': {
                'icon': '🧘',
                'exercises': ['Планка', 'Кобра', 'Собака мордой вниз', 'Воин', 'Дерево'],
                'goals': ['Гибкость', 'Расслабление', 'Улучшение осанки']
            },
            'Функциональный тренинг': {
                'icon': '⚡',
                'exercises': ['Берпи', 'Прыжки на тумбу', 'Гребля', 'Фермерская прогулка', 'Толчки санок'],
                'goals': ['Общая физическая подготовка', 'Функциональная сила', 'Выносливость']
            },
            'Кроссфит': {
                'icon': '🔥',
                'exercises': ['Трастеры', 'Подтягивания', 'Отжимания', 'Становая тяга', 'Бег'],
                'goals': ['Всестороннее развитие', 'Соревновательная подготовка', 'Высокая интенсивность']
            }
        }
        
        # База тренировочных программ
        self.training_programs = {
            'Силовые тренировки': [
                {
                    'name': 'Новичок в силовых',
                    'level': 'Начальный',
                    'description': 'Базовая программа для развития силы',
                    'exercises': ['Приседания 3x8', 'Жим лежа 3x8', 'Тяга штанги 3x8', 'Планка 3x30сек'],
                    'video_link': 'https://www.youtube.com/watch?v=example1'
                },
                {
                    'name': 'Интенсивный набор массы',
                    'level': 'Продвинутый',
                    'description': 'Программа для быстрого набора мышечной массы',
                    'exercises': ['Жим лежа 4x6', 'Становая тяга 3x5', 'Жим гантелей 3x10', 'Подтягивания 3xмакс'],
                    'video_link': 'https://www.youtube.com/watch?v=example2'
                }
            ],
            'Бег/Кардио': [
                {
                    'name': 'Старт бегуна',
                    'level': 'Начальный',
                    'description': 'Программа для начинающих бегунов',
                    'exercises': ['Интервальный бег 20 мин', 'Растяжка 10 мин', 'Силовые упражнения на ноги'],
                    'video_link': 'https://www.youtube.com/watch?v=example3'
                }
            ],
            'Йога/Пилатес': [
                {
                    'name': 'Утренняя йога',
                    'level': 'Любой',
                    'description': 'Комплекс для пробуждения и растяжки',
                    'exercises': ['Приветствие солнцу', 'Поза кошки-коровы', 'Детская поза', 'Шавасана'],
                    'video_link': 'https://www.youtube.com/watch?v=example4'
                }
            ]
        }
    
    def get_user_filename(self, username, file_type='workouts'):
        """Генерирует имя файла для пользователя"""
        user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
        return os.path.join(self.data_dir, f'{file_type}_{user_hash}.csv')
    
    def get_user_profile_filename(self, username):
        """Генерирует имя файла для профиля пользователя"""
        user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
        return os.path.join(self.data_dir, f'profile_{user_hash}.json')
    
    # Система аутентификации
    def register_user(self, username, password):
        """Регистрация нового пользователя"""
        try:
            users_file = os.path.join(self.data_dir, 'users.json')
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
            else:
                users = {}
            
            if username in users:
                return False, "Пользователь с таким именем уже существует"
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            users[username] = password_hash
            
            with open(users_file, 'w') as f:
                json.dump(users, f)
            
            # Создаем профиль с флагом, что анкета не заполнена
            profile = {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {},
                'sport_type': None,
                'questionnaire_completed': False
            }
            self.save_user_profile(username, profile)
            
            return True, "Регистрация успешна! Теперь войдите в систему."
            
        except Exception as e:
            return False, f"Ошибка регистрации: {e}"
    
    def login_user(self, username, password):
        """Авторизация пользователя"""
        try:
            users_file = os.path.join(self.data_dir, 'users.json')
            
            if not os.path.exists(users_file):
                return False, "Пользователь не найден"
            
            with open(users_file, 'r') as f:
                users = json.load(f)
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if username in users and users[username] == password_hash:
                return True, "Вход успешен"
            else:
                return False, "Неверное имя пользователя или пароль"
                
        except Exception as e:
            return False, f"Ошибка входа: {e}"
    
    def save_user_profile(self, username, profile):
        """Сохраняет профиль пользователя"""
        try:
            filename = self.get_user_profile_filename(username)
            with open(filename, 'w') as f:
                json.dump(profile, f, indent=2)
            return True
        except Exception as e:
            return False
    
    def load_user_profile(self, username):
        """Загружает профиль пользователя"""
        try:
            filename = self.get_user_profile_filename(username)
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'personal_info': {},
                    'goals': {},
                    'sport_type': None,
                    'questionnaire_completed': False
                }
        except:
            return {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {},
                'sport_type': None,
                'questionnaire_completed': False
            }
    
    def complete_questionnaire(self, username, personal_info, sport_type, goals):
        """Завершает анкету пользователя"""
        profile = self.load_user_profile(username)
        profile['personal_info'] = personal_info
        profile['sport_type'] = sport_type
        profile['goals'] = goals
        profile['questionnaire_completed'] = True
        profile['questionnaire_date'] = datetime.now().isoformat()
        
        return self.save_user_profile(username, profile)
    
    def get_recommended_trainings(self, username):
        """Возвращает рекомендуемые тренировки на основе профиля"""
        profile = self.load_user_profile(username)
        sport_type = profile.get('sport_type')
        
        if not sport_type or sport_type not in self.training_programs:
            return []
        
        return self.training_programs.get(sport_type, [])
    
    def add_workout(self, username, exercise, weight, reps, sets, notes=''):
        """Добавляет новую тренировку для пользователя"""
        try:
            new_data = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'exercise': exercise,
                'weight': float(weight),
                'reps': int(reps),
                'sets': int(sets),
                'notes': notes
            }
            
            df = pd.DataFrame([new_data])
            filename = self.get_user_filename(username)
            
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                updated_df = df
                
            updated_df.to_csv(filename, index=False)
            return True, "Тренировка успешно сохранена! 💪"
            
        except Exception as e:
            return False, f"Ошибка при сохранении: {e}"
    
    def delete_workout(self, username, workout_index):
        """Удаляет тренировку по индексу"""
        try:
            filename = self.get_user_filename(username)
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                if 0 <= workout_index < len(df):
                    df = df.drop(workout_index).reset_index(drop=True)
                    df.to_csv(filename, index=False)
                    return True, "Тренировка удалена! 🗑️"
                else:
                    return False, "Тренировка не найдена"
            return False, "Нет данных для удаления"
        except Exception as e:
            return False, f"Ошибка при удалении: {e}"
    
    def get_all_workouts(self, username):
        """Возвращает все тренировки пользователя"""
        filename = self.get_user_filename(username)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date', ascending=False)
        else:
            return pd.DataFrame(columns=['date', 'exercise', 'weight', 'reps', 'sets', 'notes'])
    
    def get_exercise_history(self, username, exercise_name):
        """Возвращает историю по конкретному упражнению пользователя"""
        df = self.get_all_workouts(username)
        if not df.empty:
            exercise_data = df[df['exercise'] == exercise_name].copy()
            return exercise_data.sort_values('date')
        return pd.DataFrame()
    
    def get_user_exercises(self, username):
        """Возвращает список всех уникальных упражнений пользователя"""
        df = self.get_all_workouts(username)
        if not df.empty:
            return df['exercise'].unique().tolist()
        return []
    
    def get_statistics(self, username):
        """Возвращает статистику тренировок пользователя"""
        df = self.get_all_workouts(username)
        if df.empty:
            return {}
        
        stats = {
            'total_workouts': len(df),
            'unique_exercises': df['exercise'].nunique(),
            'max_weight': df['weight'].max(),
            'avg_weight': df['weight'].mean(),
            'total_volume': (df['weight'] * df['reps'] * df['sets']).sum(),
            'workouts_this_month': len(df[df['date'] >= (datetime.now() - timedelta(days=30))]),
            'favorite_exercise': df['exercise'].mode().iloc[0] if not df['exercise'].mode().empty else "Нет данных",
            'last_workout': df['date'].max() if not df.empty else None,
            'workout_streak': self.calculate_streak(df)
        }
        return stats
    
    def calculate_streak(self, df):
        """Рассчитывает текущую серию тренировок подряд"""
        if df.empty:
            return 0
        
        df = df.sort_values('date', ascending=False)
        dates = df['date'].dt.date.unique()
        
        streak = 0
        current_date = datetime.now().date()
        
        for date in sorted(dates, reverse=True):
            if (current_date - date).days == streak:
                streak += 1
            else:
                break
        
        return streak
    
    def get_recommendations(self, username):
        """Генерирует рекомендации для пользователя"""
        profile = self.load_user_profile(username)
        stats = self.get_statistics(username)
        workouts = self.get_all_workouts(username)
        
        recommendations = []
        
        # Если анкета не заполнена
        if not profile.get('questionnaire_completed', False):
            recommendations.append({
                'type': 'questionnaire',
                'title': '📝 Заполните анкету',
                'description': 'Заполните анкету для персонализированных рекомендаций',
                'priority': 'high'
            })
        
        # Рекомендации на основе последней тренировки
        if not workouts.empty:
            last_workout = workouts.iloc[0]
            last_exercise = last_workout['exercise']
            last_weight = last_workout['weight']
            
            recommendations.append({
                'type': 'progress',
                'title': '📈 Продолжайте прогресс',
                'description': f'На следующей тренировке попробуйте {last_exercise} с весом {last_weight + 2.5}кг',
                'priority': 'medium'
            })
        
        # Рекомендации по регулярности
        if stats.get('last_workout'):
            days_since_last = (datetime.now() - stats['last_workout']).days
            if days_since_last > 3:
                recommendations.append({
                    'type': 'consistency',
                    'title': '⏰ Время тренировки',
                    'description': f'Прошло {days_since_last} дня с последней тренировки',
                    'priority': 'high'
                })
        
        return recommendations
    
    def get_achievements(self, username):
        """Возвращает достижения пользователя"""
        stats = self.get_statistics(username)
        workouts = self.get_all_workouts(username)
        profile = self.load_user_profile(username)
        
        achievements = []
        
        # Базовые достижения
        if stats.get('total_workouts', 0) >= 1:
            achievements.append({
                'id': 'first_workout',
                'title': '🎖️ Первая тренировка',
                'description': 'Выполнена первая тренировка',
                'icon': '🎖️',
                'unlocked': True
            })
        
        if stats.get('total_workouts', 0) >= 10:
            achievements.append({
                'id': 'dedicated',
                'title': '🔥 Посвящение',
                'description': '10 выполненных тренировок',
                'icon': '🔥',
                'unlocked': True
            })
        
        if stats.get('total_workouts', 0) >= 50:
            achievements.append({
                'id': 'veteran',
                'title': '🏅 Ветеран',
                'description': '50 выполненных тренировок',
                'icon': '🏅',
                'unlocked': True
            })
        
        if stats.get('total_workouts', 0) >= 100:
            achievements.append({
                'id': 'centurion_workouts',
                'title': '💯 Сотня тренировок',
                'description': '100 выполненных тренировок',
                'icon': '💯',
                'unlocked': True
            })
        
        if stats.get('unique_exercises', 0) >= 5:
            achievements.append({
                'id': 'versatile',
                'title': '🎯 Универсал',
                'description': 'Освоено 5 различных упражнений',
                'icon': '🎯',
                'unlocked': True
            })
        
        # Силовые достижения
        max_weight = stats.get('max_weight', 0)
        if max_weight >= 50:
            achievements.append({
                'id': 'strong_start',
                'title': '💪 Начало силы',
                'description': 'Покорен вес 50кг',
                'icon': '💪',
                'unlocked': True
            })
        
        if max_weight >= 100:
            achievements.append({
                'id': 'centurion_weight',
                'title': '🏋️‍♂️ Сотня килограммов',
                'description': 'Покорен вес 100кг',
                'icon': '🏋️‍♂️',
                'unlocked': True
            })
        
        # Достижения по регулярности
        if stats.get('workouts_this_month', 0) >= 8:
            achievements.append({
                'id': 'consistent',
                'title': '📅 Регулярность',
                'description': '8+ тренировок за месяц',
                'icon': '📅',
                'unlocked': True
            })
        
        if stats.get('workouts_this_month', 0) >= 12:
            achievements.append({
                'id': 'hardcore',
                'title': '⚡ Хардкор',
                'description': '12+ тренировок за месяц',
                'icon': '⚡',
                'unlocked': True
            })
        
        if stats.get('workout_streak', 0) >= 7:
            achievements.append({
                'id': 'weekly_streak',
                'title': '📆 Недельная серия',
                'description': '7 тренировок подряд',
                'icon': '📆',
                'unlocked': True
            })
        
        if stats.get('workout_streak', 0) >= 30:
            achievements.append({
                'id': 'monthly_streak',
                'title': '🌟 Месячная серия',
                'description': '30 тренировок подряд',
                'icon': '🌟',
                'unlocked': True
            })
        
        # Специальные достижения
        if not workouts.empty:
            # Достижение за прогресс
            first_weight = workouts.iloc[-1]['weight'] if len(workouts) > 0 else 0
            last_weight = workouts.iloc[0]['weight']
            if last_weight - first_weight >= 20:
                achievements.append({
                    'id': 'progress_master',
                    'title': '🚀 Мастер прогресса',
                    'description': 'Увеличение веса на 20+ кг',
                    'icon': '🚀',
                    'unlocked': True
                })
        
        # Достижение за заполнение анкеты
        if profile.get('questionnaire_completed', False):
            achievements.append({
                'id': 'questionnaire_complete',
                'title': '📝 Анкета заполнена',
                'description': 'Вы заполнили свою анкету',
                'icon': '📝',
                'unlocked': True
            })
        
        # Достижение за спорт
        if profile.get('sport_type'):
            sport_icon = self.sport_types.get(profile['sport_type'], {}).get('icon', '🏆')
            achievements.append({
                'id': 'sport_chosen',
                'title': f'{sport_icon} {profile["sport_type"]}',
                'description': f'Выбран вид спорта: {profile["sport_type"]}',
                'icon': sport_icon,
                'unlocked': True
            })
        
        # Новые достижения
        if stats.get('total_volume', 0) >= 10000:
            achievements.append({
                'id': 'volume_king',
                'title': '📊 Король объема',
                'description': '10,000+ кг общего объема',
                'icon': '📊',
                'unlocked': True
            })
        
        return achievements

# Инициализация приложения
app = FitnessApp()

# Система аутентификации
def initialize_session_state():
    if 'current_user' not in st.session_state:
        st.session_state.current_user = ""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False
    if 'show_questionnaire' not in st.session_state:
        st.session_state.show_questionnaire = False
    # Для навигации по страницам через кнопки
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Главная"

initialize_session_state()

# Страница входа/регистрации
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">💪 Фитнес Трекер Pro</h1>', unsafe_allow_html=True)
    
    if st.session_state.show_login:
        # Форма входа
        with st.form("login_form"):
            st.subheader("🔐 Вход в систему")
            
            login_username = st.text_input("Логин:", placeholder="Введите ваш логин")
            login_password = st.text_input("Пароль:", type="password", placeholder="Введите ваш пароль")
            
            col1, col2 = st.columns(2)
            with col1:
                login_submitted = st.form_submit_button("Войти", use_container_width=True)
            with col2:
                if st.form_submit_button("Регистрация", use_container_width=True):
                    st.session_state.show_login = False
                    st.session_state.show_registration = True
                    st.rerun()
            
            if login_submitted and login_username and login_password:
                success, message = app.login_user(login_username, login_password)
                if success:
                    st.session_state.current_user = login_username
                    st.session_state.authenticated = True
                    
                    # Проверяем, заполнена ли анкета
                    profile = app.load_user_profile(login_username)
                    if not profile.get('questionnaire_completed', False):
                        st.session_state.show_questionnaire = True
                    
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    elif st.session_state.show_registration:
        # Форма регистрации
        with st.form("register_form"):
            st.subheader("📝 Регистрация")
            
            reg_username = st.text_input("Логин:", placeholder="Придумайте логин")
            reg_password = st.text_input("Пароль:", type="password", placeholder="Придумайте пароль")
            reg_confirm = st.text_input("Подтвердите пароль:", type="password", placeholder="Повторите пароль")
            
            col1, col2 = st.columns(2)
            with col1:
                reg_submitted = st.form_submit_button("Зарегистрироваться", use_container_width=True)
            with col2:
                if st.form_submit_button("Назад к входу", use_container_width=True):
                    st.session_state.show_login = True
                    st.session_state.show_registration = False
                    st.rerun()
            
            if reg_submitted:
                if not reg_username or not reg_password:
                    st.error("❌ Заполните все поля")
                elif reg_password != reg_confirm:
                    st.error("❌ Пароли не совпадают")
                else:
                    success, message = app.register_user(reg_username, reg_password)
                    if success:
                        st.success("✅ " + message)
                        st.session_state.show_login = True
                        st.session_state.show_registration = False
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

# Анкета после регистрации/первого входа
elif st.session_state.show_questionnaire:
    st.markdown('<h1 class="main-header">📝 Давайте познакомимся!</h1>', unsafe_allow_html=True)
    
    with st.form("questionnaire_form"):
        st.subheader("📊 Личные данные")
        
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Рост (см):", min_value=100, max_value=250, value=170)
            age = st.number_input("Возраст:", min_value=10, max_value=100, value=25)
        with col2:
            weight = st.number_input("Текущий вес (кг):", min_value=30, max_value=200, value=70)
            gender = st.selectbox("Пол:", ["Мужской", "Женский"])
        
        st.subheader("🎯 Ваши цели")
        target_weight = st.number_input("Желаемый рабочий вес (кг):", min_value=0, value=80)
        primary_goal = st.selectbox("Основная цель:", 
                                  ["Увеличение силы", "Снижение веса", "Набор мышечной массы", 
                                   "Улучшение выносливости", "Общее оздоровление"])
        
        st.subheader("🏆 Вид спорта")
        st.write("Выберите основной вид активности:")
        
        # Отображение видов спорта с иконками
        sport_cols = st.columns(3)
        sport_options = list(app.sport_types.keys())
        
        selected_sport = st.selectbox(
            "Каким видом спорта вы занимаетесь?",
            sport_options,
            format_func=lambda x: f"{app.sport_types[x]['icon']} {x}"
        )
        
        if selected_sport:
            st.info(f"🎯 **{selected_sport}**: {', '.join(app.sport_types[selected_sport]['goals'][:2])}")
        
        if st.form_submit_button("✅ Сохранить анкету", use_container_width=True):
            personal_info = {
                'height': height,
                'weight': weight,
                'age': age,
                'gender': gender
            }
            
            goals = {
                'target_weight': target_weight,
                'primary_goal': primary_goal
            }
            
            if app.complete_questionnaire(st.session_state.current_user, personal_info, selected_sport, goals):
                st.session_state.show_questionnaire = False
                st.success("✅ Анкета успешно сохранена!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Ошибка сохранения анкеты")

else:
    # ОСНОВНОЕ ПРИЛОЖЕНИЕ (после входа)
    
    # Загрузка профиля пользователя
    user_profile = app.load_user_profile(st.session_state.current_user)
    
    # Отображение текущего пользователя
    st.sidebar.markdown(f'<div class="user-card">👤 Пользователь: <b>{st.session_state.current_user}</b></div>', unsafe_allow_html=True)
    
    # Показываем спорт пользователя
    if user_profile.get('sport_type'):
        sport_info = app.sport_types.get(user_profile['sport_type'], {})
        st.sidebar.markdown(f'<div class="sport-icon">{sport_info.get("icon", "🏆")} {user_profile["sport_type"]}</div>', unsafe_allow_html=True)
    
    # Основная навигация - теперь используем состояние для переключения
    with st.sidebar:
        st.title("Навигация")
        
        # Обновляем текущую страницу при выборе в радио
        page = st.radio(
            "Выберите раздел:",
            ["📊 Главная", "👤 Мой профиль", "➕ Новая тренировка", "📋 Мои тренировки", 
             "📈 Анализ прогресса", "🤖 Умные прогнозы", "🏆 Достижения", "🔄 Демо-данные"],
            index=["📊 Главная", "👤 Мой профиль", "➕ Новая тренировка", "📋 Мои тренировки", 
                   "📈 Анализ прогресса", "🤖 Умные прогнозы", "🏆 Достижения", "🔄 Демо-данные"].index(st.session_state.current_page)
        )
        
        # Обновляем состояние при изменении радио
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Статистика")
        
        stats = app.get_statistics(st.session_state.current_user)
        if stats:
            st.metric("Всего тренировок", stats['total_workouts'])
            st.metric("Упражнений", stats['unique_exercises'])
            st.metric("Серия", f"{stats.get('workout_streak', 0)} дней")
        
        st.markdown("---")
        if st.button("✏️ Редактировать анкету"):
            st.session_state.show_questionnaire = True
            st.rerun()
        
        if st.button("🚪 Выйти"):
            st.session_state.authenticated = False
            st.session_state.current_user = ""
            st.rerun()

    # Главная страница
    if st.session_state.current_page == "📊 Главная":
        st.markdown(f'<h2 class="sub-header">🏠 Добро пожаловать, {st.session_state.current_user}!</h2>', unsafe_allow_html=True)
        
        # Быстрая статистика
        stats = app.get_statistics(st.session_state.current_user)
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего тренировок", stats['total_workouts'])
            with col2:
                st.metric("Упражнений", stats['unique_exercises'])
            with col3:
                st.metric("Макс. вес", f"{stats['max_weight']:.1f} кг")
            with col4:
                st.metric("Серия", f"{stats.get('workout_streak', 0)} дней")
        
        # Рекомендации тренировок на основе вида спорта
        st.markdown("### 🎯 Рекомендуемые тренировки")
        
        if user_profile.get('sport_type'):
            sport_type = user_profile['sport_type']
            recommended_trainings = app.get_recommended_trainings(st.session_state.current_user)
            
            if recommended_trainings:
                for training in recommended_trainings[:2]:  # Показываем 2 программы
                    with st.container():
                        st.markdown(f"""
                        <div class="training-card">
                            <h4>🏋️ {training['name']} ({training['level']})</h4>
                            <p>{training['description']}</p>
                            <p><strong>Упражнения:</strong> {', '.join(training['exercises'])}</p>
                            <a href="{training.get('video_link', '#')}" target="_blank">📹 Посмотреть видео</a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info(f"💡 Для {sport_type} мы готовим программы тренировок. Скоро они появятся!")
        else:
            st.info("💡 Заполните анкету, чтобы получить персональные рекомендации тренировок!")
        
        # Персональные рекомендации
        recommendations = app.get_recommendations(st.session_state.current_user)
        
        if recommendations:
            st.markdown("### 💡 Персональные рекомендации")
            for rec in recommendations[:3]:
                priority_color = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{priority_color} {rec['title']}</h4>
                    <p>{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Последние тренировки
        st.markdown("### 📋 Последние тренировки")
        workouts = app.get_all_workouts(st.session_state.current_user)
        
        if not workouts.empty:
            recent_workouts = workouts.head(3)
            for _, workout in recent_workouts.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{workout['exercise']}**")
                        if workout['notes']:
                            st.caption(f"💬 {workout['notes']}")
                    with col2:
                        st.markdown(f"**{workout['weight']}кг** × {workout['reps']} × {workout['sets']}")
                    with col3:
                        st.caption(workout['date'].strftime('%d.%m.%Y'))
                    st.markdown("---")
            
            if st.button("📋 Показать все тренировки"):
                st.session_state.current_page = "📋 Мои тренировки"
                st.rerun()
        else:
            st.info("🎯 У вас пока нет тренировок. Добавьте первую!")
        
        # Быстрые действия (РАБОЧИЕ КНОПКИ)
        st.markdown("### ⚡ Быстрые действия")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Добавить тренировку", use_container_width=True):
                st.session_state.current_page = "➕ Новая тренировка"
                st.rerun()
        with col2:
            if st.button("📊 Анализ прогресса", use_container_width=True):
                st.session_state.current_page = "📈 Анализ прогресса"
                st.rerun()
        with col3:
            if st.button("🤖 Умные прогнозы", use_container_width=True):
                st.session_state.current_page = "🤖 Умные прогнозы"
                st.rerun()

    # Мой профиль
    elif st.session_state.current_page == "👤 Мой профиль":
        st.markdown('<h2 class="sub-header">👤 Мой профиль</h2>', unsafe_allow_html=True)
        
        with st.form("profile_form"):
            st.subheader("📏 Личные параметры")
            
            col1, col2 = st.columns(2)
            with col1:
                height = st.number_input("Рост (см):", min_value=100, max_value=250, 
                                       value=user_profile.get('personal_info', {}).get('height', 170))
                age = st.number_input("Возраст:", min_value=10, max_value=100, 
                                    value=user_profile.get('personal_info', {}).get('age', 25))
            with col2:
                weight = st.number_input("Вес (кг):", min_value=30, max_value=200, 
                                       value=user_profile.get('personal_info', {}).get('weight', 70))
                gender = st.selectbox("Пол:", ["Мужской", "Женский"], 
                                    index=0 if user_profile.get('personal_info', {}).get('gender') == "Мужской" else 1)
            
            st.subheader("🎯 Мои цели")
            target_weight = st.number_input("Целевой вес в упражнениях (кг):", min_value=0, 
                                          value=user_profile.get('goals', {}).get('target_weight', 0))
            
            st.subheader("🏆 Вид спорта")
            current_sport = user_profile.get('sport_type', 'Силовые тренировки')
            sport_options = list(app.sport_types.keys())
            new_sport = st.selectbox(
                "Основной вид активности:",
                sport_options,
                index=sport_options.index(current_sport) if current_sport in sport_options else 0,
                format_func=lambda x: f"{app.sport_types[x]['icon']} {x}"
            )
            
            if st.form_submit_button("💾 Сохранить изменения", use_container_width=True):
                personal_info = {
                    'height': height,
                    'weight': weight,
                    'age': age,
                    'gender': gender
                }
                goals = {
                    'target_weight': target_weight,
                    'primary_goal': user_profile.get('goals', {}).get('primary_goal', 'Увеличение силы')
                }
                
                # Обновляем анкету
                if app.complete_questionnaire(st.session_state.current_user, personal_info, new_sport, goals):
                    st.success("✅ Профиль успешно обновлен!")
                    st.rerun()
                else:
                    st.error("❌ Ошибка сохранения профиля")

    # Добавление тренировки
    elif st.session_state.current_page == "➕ Новая тренировка":
        st.markdown(f'<h2 class="sub-header">➕ Новая тренировка</h2>', unsafe_allow_html=True)
        
        # Получаем рекомендуемые упражнения для вида спорта
        sport_type = user_profile.get('sport_type', 'Силовые тренировки')
        recommended_exercises = app.sport_types.get(sport_type, {}).get('exercises', [])
        
        if 'workout_data' not in st.session_state:
            st.session_state.workout_data = {
                'exercise': '',
                'weight': 50.0,
                'reps': 8,
                'sets': 4,
                'notes': ''
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Основные данные")
            
            st.write(f"**Рекомендуемые для {sport_type}:**")
            preset_cols = st.columns(min(5, len(recommended_exercises)))
            
            for i, exercise in enumerate(recommended_exercises[:5]):
                with preset_cols[i % 5]:
                    if st.button(exercise[:10], key=f"preset_{i}", help=exercise):
                        st.session_state.workout_data['exercise'] = exercise
                        st.rerun()
            
            exercise = st.text_input(
                "Упражнение 🏋️",
                value=st.session_state.workout_data['exercise'],
                placeholder="Введите название упражнения...",
                key="exercise_input"
            )
            
            weight = st.number_input(
                "Вес (кг) ⚖️", 
                min_value=0.0, 
                step=0.5,
                value=st.session_state.workout_data['weight'],
                key="weight_input"
            )
        
        with col2:
            st.subheader("Параметры")
            reps = st.number_input(
                "Количество повторений 🔁", 
                min_value=1, 
                step=1,
                value=st.session_state.workout_data['reps'],
                key="reps_input"
            )
            
            sets = st.number_input(
                "Количество подходов 📊", 
                min_value=1, 
                step=1,
                value=st.session_state.workout_data['sets'],
                key="sets_input"
            )
        
        notes = st.text_area(
            "Заметки к тренировке 📝", 
            value=st.session_state.workout_data['notes'],
            placeholder="Опишите как прошла тренировка...",
            height=100,
            key="notes_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Сохранить тренировку", use_container_width=True, type="primary"):
                if exercise and weight > 0 and reps > 0 and sets > 0:
                    success, message = app.add_workout(
                        st.session_state.current_user, exercise, weight, reps, sets, notes
                    )
                    if success:
                        st.success(message)
                        st.balloons()
                        st.session_state.workout_data = {'exercise': '', 'weight': 50.0, 'reps': 8, 'sets': 4, 'notes': ''}
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("❌ Заполните все обязательные поля!")
        
        with col2:
            if st.button("🏠 На главную", use_container_width=True):
                st.session_state.current_page = "📊 Главная"
                st.rerun()

    # Мои тренировки
    elif st.session_state.current_page == "📋 Мои тренировки":
        st.markdown(f'<h2 class="sub-header">📋 Мои тренировки</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.info("📝 У вас пока нет тренировок. Добавьте первую!")
        else:
            st.info(f"🎯 Всего тренировок: {len(df)}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                search_exercise = st.text_input("🔍 Поиск по упражнению:", placeholder="Введите название упражнения...")
            with col2:
                show_count = st.selectbox("Показывать:", [10, 25, 50, "Все"])
            
            display_df = df.copy()
            if search_exercise:
                display_df = display_df[display_df['exercise'].str.contains(search_exercise, case=False, na=False)]
            
            if show_count != "Все":
                display_df = display_df.head(show_count)
            
            display_df = display_df.reset_index(drop=True)
            display_df['date'] = display_df['date'].dt.strftime('%d.%m.%Y %H:%M')
            display_df['volume'] = display_df['weight'] * display_df['reps'] * display_df['sets']
            
            for idx, workout in display_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{workout['exercise']}**")
                        if workout['notes']:
                            st.caption(f"💬 {workout['notes']}")
                        st.caption(f"📅 {workout['date']}")
                    with col2:
                        st.markdown(f"**{workout['weight']}кг** × {workout['reps']} × {workout['sets']}")
                    with col3:
                        st.markdown(f"**Объем:** {workout['volume']:.0f} кг")
                    with col4:
                        if st.button("🗑️", key=f"delete_{idx}", help="Удалить тренировку"):
                            success, message = app.delete_workout(st.session_state.current_user, idx)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    st.markdown("---")

    # Анализ прогресса
    elif st.session_state.current_page == "📈 Анализ прогресса":
        st.markdown(f'<h2 class="sub-header">📈 Анализ прогресса</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.warning("📝 Нет данных для анализа. Добавьте несколько тренировок!")
        else:
            exercises = app.get_user_exercises(st.session_state.current_user)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_exercise = st.selectbox("Выберите упражнение для анализа:", exercises, index=0)
                
                if selected_exercise:
                    exercise_data = app.get_exercise_history(st.session_state.current_user, selected_exercise)
                    
                    if not exercise_data.empty:
                        st.markdown("### 📊 Статистика")
                        max_weight = exercise_data['weight'].max()
                        min_weight = exercise_data['weight'].min()
                        avg_weight = exercise_data['weight'].mean()
                        workouts_count = len(exercise_data)
                        total_volume = (exercise_data['weight'] * exercise_data['reps'] * exercise_data['sets']).sum()
                        progress = max_weight - min_weight
                        
                        st.metric("Максимальный вес", f"{max_weight:.1f} кг")
                        st.metric("Средний вес", f"{avg_weight:.1f} кг")
                        st.metric("Количество тренировок", workouts_count)
                        st.metric("Общий объем", f"{total_volume:.0f} кг")
                        
                        if progress > 0:
                            st.success(f"📈 Общий прогресс: +{progress:.1f} кг")
                        else:
                            st.info("📊 Прогресс: 0 кг")
            
            with col2:
                if selected_exercise and not exercise_data.empty:
                    st.markdown("### 📈 График прогресса")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    ax1.plot(exercise_data['date'], exercise_data['weight'], 'o-', linewidth=2, markersize=6, color='#1f77b4')
                    ax1.set_title(f'Прогресс в упражнении: {selected_exercise}', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Вес (кг)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    exercise_data['volume'] = exercise_data['weight'] * exercise_data['reps'] * exercise_data['sets']
                    ax2.plot(exercise_data['date'], exercise_data['volume'], 's-', linewidth=2, markersize=6, color='#ff7f0e')
                    ax2.set_title(f'Объем тренировок: {selected_exercise}', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Дата', fontsize=12)
                    ax2.set_ylabel('Объем (кг)', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

    # Умные прогнозы (ТОЛЬКО 1 МЕСЯЦ + РЕАЛЬНЫЕ ДАННЫЕ)
    elif st.session_state.current_page == "🤖 Умные прогнозы":
        st.markdown(f'<h2 class="sub-header">🤖 Умные прогнозы</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if len(df) < 5:  # Увеличили до 5 тренировок для лучшего прогноза
            st.warning("""
            ⚠️ Для точных прогнозов нужно минимум 5 тренировок по одному упражнению.
            
            **Что делать:**
            1. Добавьте больше тренировок через раздел "➕ Новая тренировка"
            2. Или создайте демо-данные через раздел "🔄 Демо-данные"
            """)
        else:
            exercises = app.get_user_exercises(st.session_state.current_user)
            selected_exercise = st.selectbox(
                "Выберите упражнение для прогноза:",
                exercises,
                key="ml_exercise"
            )
            
            if selected_exercise:
                exercise_data = app.get_exercise_history(st.session_state.current_user, selected_exercise)
                
                if len(exercise_data) >= 5:
                    # Подготовка данных с реальными примерами
                    exercise_data = exercise_data.copy()
                    exercise_data = exercise_data.sort_values('date')
                    exercise_data['days_passed'] = (exercise_data['date'] - exercise_data['date'].min()).dt.days
                    
                    # Используем полиномиальную регрессию для лучшего прогноза
                    X = exercise_data[['days_passed']].values
                    y = exercise_data['weight'].values
                    
                    # Создаем полиномиальные признаки
                    X_poly = np.column_stack([X, X**2])  # Добавляем квадратичный член
                    
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    
                    # Прогноз на 1 месяц
                    last_day = exercise_data['days_passed'].max()
                    days_in_month = 30
                    future_day = last_day + days_in_month
                    
                    # Создаем полиномиальные признаки для прогноза
                    future_X = np.array([[future_day, future_day**2]])
                    predicted_weight = model.predict(future_X)[0]
                    
                    # Текущие показатели
                    current_weight = exercise_data['weight'].iloc[-1]
                    progress_rate = (current_weight - exercise_data['weight'].iloc[0]) / len(exercise_data) if len(exercise_data) > 0 else 0
                    
                    # Отображение прогноза
                    st.markdown("### 📊 Прогноз на 1 месяц")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        delta = predicted_weight - current_weight
                        st.metric(
                            "Прогноз через 1 месяц",
                            f"{predicted_weight:.1f} кг",
                            delta=f"{delta:.1f} кг",
                            delta_color="normal" if delta > 0 else "off"
                        )
                    
                    with col2:
                        st.metric(
                            "Текущий вес",
                            f"{current_weight:.1f} кг",
                            delta=f"{progress_rate:.2f} кг/тренировка" if progress_rate > 0 else "0 кг/тренировка"
                        )
                    
                    # Детальный анализ
                    st.markdown("### 📈 Детальный анализ")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **Анализ прогресса:**
                        - Начальный вес: {exercise_data['weight'].iloc[0]:.1f} кг
                        - Текущий вес: {current_weight:.1f} кг
                        - Всего тренировок: {len(exercise_data)}
                        - Средний прирост: {progress_rate:.2f} кг/тренировка
                        """)
                    
                    with col2:
                        if delta > 0:
                            st.success(f"""
                            **Прогноз положительный!** 🎉
                            - Ожидаемый прирост: {delta:.1f} кг за месяц
                            - Рекомендуемый вес на следующей тренировке: {current_weight + 2.5:.1f} кг
                            """)
                        else:
                            st.warning("""
                            **Прогноз стабильный** ⚠️
                            - Рекомендуется увеличить интенсивность
                            - Рассмотрите изменение программы тренировок
                            """)
                    
                    # График с прогнозом
                    st.markdown("### 📊 График прогресса с прогнозом")
                    
                    # Создаем точки для прогноза
                    future_days = np.linspace(last_day, future_day, 5)
                    future_X_plot = np.column_stack([future_days, future_days**2])
                    future_predictions = model.predict(future_X_plot)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Исторические данные
                    ax.plot(exercise_data['days_passed'], exercise_data['weight'], 'o-', 
                           linewidth=2, markersize=6, label='Исторические данные', color='#1f77b4')
                    
                    # Прогноз
                    ax.plot(future_days, future_predictions, '--', 
                           linewidth=2, label='Прогноз на 1 месяц', color='#ff7f0e')
                    
                    ax.set_title(f'Прогресс и прогноз для {selected_exercise}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Дни с первой тренировки')
                    ax.set_ylabel('Вес (кг)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Рекомендации на основе реальных данных
                    st.markdown("### 💡 Рекомендации на основе ваших данных")
                    
                    if progress_rate > 0.3:
                        st.success("""
                        **Отличный прогресс!** 🚀
                        - Продолжайте текущую программу
                        - Увеличивайте вес на 2.5-5 кг каждые 2 недели
                        - Следите за восстановлением
                        """)
                    elif progress_rate > 0.1:
                        st.info("""
                        **Хороший стабильный прогресс** 📈
                        - Увеличивайте вес на 1-2.5 кг каждые 2-3 тренировки
                        - Добавьте вспомогательные упражнения
                        """)
                    else:
                        st.warning("""
                        **Прогресс замедлился** ⚡
                        - Рекомендуется изменить программу тренировок
                        - Увеличьте частоту тренировок до 3-4 раз в неделю
                        - Проверьте питание и сон
                        """)
                else:
                    st.warning(f"Для упражнения '{selected_exercise}' нужно минимум 5 тренировок для точного прогноза. Сейчас: {len(exercise_data)}")

    # Достижения (ПОЛНАЯ ВЕРСИЯ)
    elif st.session_state.current_page == "🏆 Достижения":
        st.markdown(f'<h2 class="sub-header">🏆 Мои достижения</h2>', unsafe_allow_html=True)
        
        achievements = app.get_achievements(st.session_state.current_user)
        stats = app.get_statistics(st.session_state.current_user)
        
        if achievements:
            unlocked = [a for a in achievements if a.get('unlocked', False)]
            locked = [a for a in achievements if not a.get('unlocked', False)]
            
            st.success(f"🎉 У вас {len(unlocked)} из {len(achievements)} достижений!")
            
            st.markdown("### 🏆 Полученные достижения")
            if unlocked:
                cols = st.columns(3)
                for i, achievement in enumerate(unlocked):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="achievement-card">
                            <h3>{achievement['icon']}</h3>
                            <h4>{achievement['title']}</h4>
                            <p>{achievement['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Показываем ближайшие цели
            if stats:
                st.markdown("### 🎯 Ближайшие цели")
                goals_data = []
                
                if stats['total_workouts'] < 10:
                    goals_data.append(["🔥 Посвящение", f"{stats['total_workouts']}/10", "10 тренировок"])
                elif stats['total_workouts'] < 50:
                    goals_data.append(["🏅 Ветеран", f"{stats['total_workouts']}/50", "50 тренировок"])
                elif stats['total_workouts'] < 100:
                    goals_data.append(["💯 Сотня тренировок", f"{stats['total_workouts']}/100", "100 тренировок"])
                
                max_weight = stats.get('max_weight', 0)
                if max_weight < 50:
                    goals_data.append(["💪 Начало силы", f"{max_weight:.1f}/50", f"{50 - max_weight:.1f} кг"])
                elif max_weight < 100:
                    goals_data.append(["🏋️‍♂️ Сотня килограммов", f"{max_weight:.1f}/100", f"{100 - max_weight:.1f} кг"])
                
                if stats.get('workout_streak', 0) < 7:
                    goals_data.append(["📆 Недельная серия", f"{stats['workout_streak']}/7", f"{7 - stats['workout_streak']} дней"])
                
                if goals_data:
                    goals_df = pd.DataFrame(goals_data, columns=['Достижение', 'Прогресс', 'Осталось'])
                    st.dataframe(goals_df, use_container_width=True, hide_index=True)
                else:
                    st.success("🎊 Все основные цели достигнуты! Вы настоящий чемпион! 🏆")
        else:
            st.info("""
            **Начните тренироваться чтобы получить достижения!** 🏋️
            
            **Доступные достижения:**
            🎖️ **Первая тренировка** - Выполните первую тренировку
            🔥 **Посвящение** - 10 тренировок
            🏅 **Ветеран** - 50 тренировок
            💯 **Сотня тренировок** - 100 тренировок
            🎯 **Универсал** - 5 различных упражнений
            💪 **Начало силы** - Покорите вес 50кг
            🏋️‍♂️ **Сотня килограммов** - Покорите вес 100кг
            📅 **Регулярность** - 8+ тренировок за месяц
            ⚡ **Хардкор** - 12+ тренировок за месяц
            📆 **Недельная серия** - 7 тренировок подряд
            🌟 **Месячная серия** - 30 тренировок подряд
            🚀 **Мастер прогресса** - Увеличение веса на 20+ кг
            📝 **Анкета заполнена** - Заполнение анкеты
            📊 **Король объема** - 10,000+ кг общего объема
            """)

    # Демо-данные
    elif st.session_state.current_page == "🔄 Демо-данные":
        st.markdown(f'<h2 class="sub-header">🔄 Демо-данные</h2>', unsafe_allow_html=True)
        
        st.info("""
        **Демо-данные** помогут вам протестировать все функции приложения.
        Будут созданы реалистичные данные за последние 2 месяца.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Создать демо-данные", use_container_width=True, type="primary"):
                # Генерация уникальных данных на основе профиля
                profile = app.load_user_profile(st.session_state.current_user)
                sport_type = profile.get('sport_type', 'Силовые тренировки')
                
                demo_workouts = []
                base_date = datetime.now() - timedelta(days=60)
                
                # Получаем упражнения для выбранного вида спорта
                exercises = app.sport_types.get(sport_type, {}).get('exercises', ['Приседания', 'Жим лежа', 'Становая тяга'])
                
                # Генерируем прогрессивные тренировки
                for i, exercise in enumerate(exercises):
                    for j in range(6):  # 6 тренировок на каждое упражнение
                        date = base_date + timedelta(days=j*10 + i*2)
                        
                        # Базовый вес зависит от вида спорта
                        if sport_type == 'Силовые тренировки':
                            base_weight = [60, 50, 80, 40, 70][i % 5] + j * 5
                            reps = 8 if j < 4 else 6
                        elif sport_type == 'Бег/Кардио':
                            base_weight = 0  # Для кардио вес не важен
                            reps = [20, 25, 30, 35, 25, 30][j]
                        elif sport_type == 'Йога/Пилатес':
                            base_weight = 0
                            reps = [10, 12, 15, 12, 15, 15][j]
                        else:
                            base_weight = 40 + j * 3
                            reps = 10
                        
                        sets = 4 if sport_type in ['Силовые тренировки', 'Кроссфит'] else 3
                        
                        demo_workouts.append((
                            date.strftime('%Y-%m-%d %H:%M:%S'),
                            exercise,
                            base_weight,
                            reps,
                            sets,
                            f"{sport_type} - {exercise} - неделя {j+1}"
                        ))
                
                # Сохраняем все демо-тренировки
                count = 0
                for workout in demo_workouts:
                    success, _ = app.add_workout(st.session_state.current_user, workout[1], workout[2], workout[3], workout[4], workout[5])
                    if success:
                        count += 1
                
                st.success(f"✅ Создано {count} демо-тренировок для {sport_type}!")
                st.balloons()
                
                st.markdown(f"""
                ### 📊 Что было создано:
                - **Реалистичная история** тренировок за 2 месяца
                - **{len(exercises)} различных упражнения** для {sport_type}
                - **Постепенный прогресс** в весах/повторениях
                - **Готовые данные** для тестирования всех функций
                """)
        
        with col2:
            if st.button("🗑️ Очистить мои данные", type="secondary"):
                filename = app.get_user_filename(st.session_state.current_user)
                if os.path.exists(filename):
                    os.remove(filename)
                    st.success("✅ Ваши данные очищены!")
                    st.rerun()
            
            st.warning("""
            ⚠️ **Внимание!**
            При создании демо-данных:
            - Существующие тренировки будут сохранены
            - Добавятся новые демо-тренировки
            - Данные генерируются на основе вашего вида спорта
            """)

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💪 <strong>Фитнес Трекер Pro v5.0</strong> | Персональный тренер в вашем кармане</p>
</div>
""", unsafe_allow_html=True)
