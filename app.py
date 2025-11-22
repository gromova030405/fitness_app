import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import hashlib
import json

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
    }
    .achievement-card {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class FitnessApp:
    def __init__(self):
        self.data_dir = 'user_data'
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Создает папку для данных если её нет"""
        os.makedirs(self.data_dir, exist_ok=True)
    
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
            
            # Загружаем существующих пользователей
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    users = json.load(f)
            else:
                users = {}
            
            # Проверяем, существует ли пользователь
            if username in users:
                return False, "Пользователь с таким именем уже существует"
            
            # Хешируем пароль
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            users[username] = password_hash
            
            # Сохраняем пользователей
            with open(users_file, 'w') as f:
                json.dump(users, f)
            
            # Создаем пустой профиль
            profile = {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {}
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
                # Создаем пустой профиль
                return {
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'personal_info': {},
                    'goals': {}
                }
        except:
            return {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {}
            }
    
    def update_personal_info(self, username, personal_info):
        """Обновляет личные параметры пользователя"""
        profile = self.load_user_profile(username)
        profile['personal_info'] = personal_info
        profile['updated_at'] = datetime.now().isoformat()
        return self.save_user_profile(username, profile)
    
    def update_goals(self, username, goals):
        """Обновляет цели пользователя"""
        profile = self.load_user_profile(username)
        profile['goals'] = goals
        profile['updated_at'] = datetime.now().isoformat()
        return self.save_user_profile(username, profile)
    
    # Основные функции приложения
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
            'last_workout': df['date'].max() if not df.empty else None
        }
        return stats
    
    def get_recommendations(self, username):
        """Генерирует рекомендации для пользователя"""
        profile = self.load_user_profile(username)
        stats = self.get_statistics(username)
        workouts = self.get_all_workouts(username)
        
        recommendations = []
        
        # Рекомендации на основе последней тренировки
        if not workouts.empty:
            last_workout = workouts.iloc[0]
            last_exercise = last_workout['exercise']
            last_weight = last_workout['weight']
            
            recommendations.append({
                'type': 'progress',
                'title': '📈 Продолжайте прогресс',
                'description': f'На следующей тренировке попробуйте {last_exercise} с весом {last_weight + 2.5}кг',
                'priority': 'high'
            })
        
        # Рекомендации на основе целей
        if profile.get('goals', {}).get('target_weight'):
            target = profile['goals']['target_weight']
            current = stats.get('max_weight', 0)
            if current < target:
                recommendations.append({
                    'type': 'goal',
                    'title': '🎯 Двигайтесь к цели',
                    'description': f'До вашей цели {target}кг осталось {target - current:.1f}кг',
                    'priority': 'medium'
                })
        
        # Рекомендации по разнообразию
        if stats.get('unique_exercises', 0) < 3:
            recommendations.append({
                'type': 'variety',
                'title': '🔄 Добавьте разнообразия',
                'description': 'Попробуйте новые упражнения для равномерного развития',
                'priority': 'medium'
            })
        
        # Рекомендации по регулярности
        if stats.get('last_workout'):
            days_since_last = (datetime.now() - stats['last_workout']).days
            if days_since_last > 7:
                recommendations.append({
                    'type': 'consistency',
                    'title': '⏰ Время тренировки',
                    'description': f'Прошло {days_since_last} дней с последней тренировки',
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
                'description': '10 completed workouts',
                'icon': '🔥',
                'unlocked': True
            })
        
        if stats.get('total_workouts', 0) >= 50:
            achievements.append({
                'id': 'veteran',
                'title': '🏅 Ветеран',
                'description': '50 completed workouts',
                'icon': '🏅',
                'unlocked': True
            })
        
        if stats.get('unique_exercises', 0) >= 5:
            achievements.append({
                'id': 'versatile',
                'title': '🎯 Универсал',
                'description': '5 different exercises mastered',
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
                'id': 'centurion',
                'title': '💯 Сотня',
                'description': 'Покорен вес 100кг',
                'icon': '💯',
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
        
        # Специальные достижения
        if not workouts.empty:
            # Достижение за прогресс
            first_weight = workouts.iloc[-1]['weight']
            last_weight = workouts.iloc[0]['weight']
            if last_weight - first_weight >= 20:
                achievements.append({
                    'id': 'progress_master',
                    'title': '🚀 Мастер прогресса',
                    'description': 'Увеличение веса на 20+ кг',
                    'icon': '🚀',
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
                    st.success("✅ Вход успешен!")
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

else:
    # ОСНОВНОЕ ПРИЛОЖЕНИЕ (после входа)
    
    # Загрузка профиля пользователя
    user_profile = app.load_user_profile(st.session_state.current_user)
    
    # Отображение текущего пользователя
    st.sidebar.markdown(f'<div class="user-card">👤 Пользователь: <b>{st.session_state.current_user}</b></div>', unsafe_allow_html=True)
    
    # Основная навигация
    with st.sidebar:
        st.title("Навигация")
        
        page = st.radio(
            "Выберите раздел:",
            ["📊 Главная", "👤 Мой профиль", "➕ Новая тренировка", "📋 Мои тренировки", 
             "📈 Анализ прогресса", "🤖 Умные прогнозы", "🏆 Достижения", "🔄 Демо-данные"]
        )
        
        st.markdown("---")
        st.markdown("### Статистика")
        
        stats = app.get_statistics(st.session_state.current_user)
        if stats:
            st.metric("Всего тренировок", stats['total_workouts'])
            st.metric("Упражнений", stats['unique_exercises'])
            st.metric("Макс. вес", f"{stats['max_weight']:.1f} кг")
        
        st.markdown("---")
        if st.button("🚪 Выйти"):
            st.session_state.authenticated = False
            st.session_state.current_user = ""
            st.rerun()

    # Главная страница
    if page == "📊 Главная":
        st.markdown(f'<h2 class="sub-header">🏠 Добро пожаловать, {st.session_state.current_user}!</h2>', unsafe_allow_html=True)
        
        # Быстрая статистика
        stats = app.get_statistics(st.session_state.current_user)
        profile = app.load_user_profile(st.session_state.current_user)
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего тренировок", stats['total_workouts'])
            with col2:
                st.metric("Упражнений", stats['unique_exercises'])
            with col3:
                st.metric("Макс. вес", f"{stats['max_weight']:.1f} кг")
            with col4:
                st.metric("За месяц", stats['workouts_this_month'])
        
        # Рекомендации
        st.markdown("### 💡 Рекомендуемые тренировки")
        recommendations = app.get_recommendations(st.session_state.current_user)
        
        if recommendations:
            for rec in recommendations[:3]:  # Показываем 3 рекомендации
                priority_color = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{priority_color} {rec['title']}</h4>
                    <p>{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Добавьте тренировки чтобы получить персональные рекомендации!")
        
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
        else:
            st.info("🎯 У вас пока нет тренировок. Добавьте первую!")
        
        # Быстрые действия
        st.markdown("### ⚡ Быстрые действия")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Добавить тренировку", use_container_width=True):
                st.session_state.page = "➕ Новая тренировка"
                st.rerun()
        with col2:
            if st.button("📊 Анализ прогресса", use_container_width=True):
                st.session_state.page = "📈 Анализ прогресса"
                st.rerun()
        with col3:
            if st.button("🏆 Мои достижения", use_container_width=True):
                st.session_state.page = "🏆 Достижения"
                st.rerun()

    # Мой профиль
    elif page == "👤 Мой профиль":
        st.markdown(f'<h2 class="sub-header">👤 Мой профиль</h2>', unsafe_allow_html=True)
        
        with st.form("profile_form"):
            st.subheader("📏 Личные параметры")
            
            col1, col2 = st.columns(2)
            with col1:
                height = st.number_input("Рост (см):", min_value=100, max_value=250, value=user_profile.get('personal_info', {}).get('height', 170))
                age = st.number_input("Возраст:", min_value=10, max_value=100, value=user_profile.get('personal_info', {}).get('age', 25))
            with col2:
                weight = st.number_input("Вес (кг):", min_value=30, max_value=200, value=user_profile.get('personal_info', {}).get('weight', 70))
                gender = st.selectbox("Пол:", ["Мужской", "Женский"], index=0 if user_profile.get('personal_info', {}).get('gender') == "Мужской" else 1)
            
            st.subheader("🎯 Мои цели")
            col1, col2 = st.columns(2)
            with col1:
                target_weight = st.number_input("Целевой вес в упражнениях (кг):", min_value=0, value=user_profile.get('goals', {}).get('target_weight', 0))
            with col2:
                target_workouts = st.number_input("Целевое количество тренировок в неделю:", min_value=1, max_value=7, value=user_profile.get('goals', {}).get('target_workouts', 3))
            
            if st.form_submit_button("💾 Сохранить профиль", use_container_width=True):
                personal_info = {
                    'height': height,
                    'weight': weight,
                    'age': age,
                    'gender': gender
                }
                goals = {
                    'target_weight': target_weight,
                    'target_workouts': target_workouts
                }
                
                if app.update_personal_info(st.session_state.current_user, personal_info) and app.update_goals(st.session_state.current_user, goals):
                    st.success("✅ Профиль успешно сохранен!")
                else:
                    st.error("❌ Ошибка сохранения профиля")

    # Добавление тренировки (остается без изменений)
    elif page == "➕ Новая тренировка":
        st.markdown(f'<h2 class="sub-header">➕ Новая тренировка</h2>', unsafe_allow_html=True)
        
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
            
            st.write("**Быстрый выбор:**")
            preset_cols = st.columns(5)
            preset_exercises = ["Жим лежа", "Приседания", "Становая тяга", "Тяга к поясу", "Жим стоя"]
            
            for i, preset in enumerate(preset_exercises):
                with preset_cols[i]:
                    if st.button(preset, key=f"preset_{i}"):
                        st.session_state.workout_data['exercise'] = preset
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
            if st.button("🧹 Очистить форму", use_container_width=True):
                st.session_state.workout_data = {'exercise': '', 'weight': 50.0, 'reps': 8, 'sets': 4, 'notes': ''}
                st.rerun()

    # Мои тренировки (остается без изменений)
    elif page == "📋 Мои тренировки":
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

    # Анализ прогресса (остается без изменений)
    elif page == "📈 Анализ прогресса":
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

    # Умные прогнозы (РАБОЧАЯ ВЕРСИЯ)
    elif page == "🤖 Умные прогнозы":
        st.markdown(f'<h2 class="sub-header">🤖 Умные прогнозы</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if len(df) < 3:
            st.warning("""
            ⚠️ Для работы умных прогнозов нужно как минимум 3 тренировки по одному упражнению.
            
            **Что делать:**
            1. Добавьте больше тренировок через раздел "➕ Новая тренировка"
            2. Или создайте демо-данные через раздел "🔄 Демо-данные"
            ""
