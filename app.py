import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os
import hashlib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Настройка страницы
st.set_page_config(
    page_title="💪 Фитнес Помощник",
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
    .training-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f9fff9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .training-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .achievement-card {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .progress-card {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sport-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
        text-align: center;
    }
    .goal-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    .weight-loss { background: #ff6b6b; color: white; }
    .muscle-gain { background: #4ecdc4; color: white; }
    .endurance { background: #45b7d1; color: white; }
    .flexibility { background: #96ceb4; color: white; }
    .health { background: #feca57; color: white; }
</style>
""", unsafe_allow_html=True)

class FitnessAssistant:
    def __init__(self):
        self.data_dir = 'user_data'
        self._ensure_data_directory()
        # Инициализация базы знаний о тренировках
        self.init_training_knowledge_base()
        # Загрузка или создание ML модели
        self.init_ml_model()
    
    def _ensure_data_directory(self):
        """Создает папку для данных если её нет"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def init_training_knowledge_base(self):
        """Инициализация базы знаний о тренировках для разных целей"""
        
        # Основные цели пользователей
        self.goals = {
            'weight_loss': {
                'name': 'Похудение',
                'icon': '⚖️',
                'color': 'weight-loss',
                'description': 'Снижение веса и уменьшение жировой массы'
            },
            'muscle_gain': {
                'name': 'Набор мышечной массы',
                'icon': '💪',
                'color': 'muscle-gain',
                'description': 'Увеличение мышечной массы и силы'
            },
            'endurance': {
                'name': 'Улучшение выносливости',
                'icon': '🏃',
                'color': 'endurance',
                'description': 'Повышение кардио-выносливости'
            },
            'flexibility': {
                'name': 'Развитие гибкости',
                'icon': '🧘',
                'color': 'flexibility',
                'description': 'Улучшение гибкости и мобильности'
            },
            'health': {
                'name': 'Общее оздоровление',
                'icon': '❤️',
                'color': 'health',
                'description': 'Улучшение общего состояния здоровья'
            }
        }
        
        # Виды физической активности
        self.activity_types = {
            'yoga': {
                'name': 'Йога',
                'icon': '🧘',
                'description': 'Практики для развития гибкости и равновесия',
                'intensity': 'Низкая',
                'calories_per_hour': 200,
                'equipment': 'Коврик'
            },
            'pilates': {
                'name': 'Пилатес',
                'icon': '🤸',
                'description': 'Упражнения для укрепления мышц кора',
                'intensity': 'Средняя',
                'calories_per_hour': 250,
                'equipment': 'Коврик, мяч'
            },
            'circuit_training': {
                'name': 'Круговые тренировки',
                'icon': '🔄',
                'description': 'Интенсивные тренировки по кругу',
                'intensity': 'Высокая',
                'calories_per_hour': 500,
                'equipment': 'Гантели, коврик'
            },
            'cardio': {
                'name': 'Кардио-тренировки',
                'icon': '🏃',
                'description': 'Тренировки для сердечно-сосудистой системы',
                'intensity': 'Средняя-Высокая',
                'calories_per_hour': 400,
                'equipment': 'Беговая дорожка, велотренажер'
            },
            'strength': {
                'name': 'Силовые тренировки',
                'icon': '🏋️',
                'description': 'Упражнения с отягощениями',
                'intensity': 'Средняя-Высокая',
                'calories_per_hour': 300,
                'equipment': 'Гантели, штанга'
            },
            'stretching': {
                'name': 'Растяжка',
                'icon': '✨',
                'description': 'Упражнения на растяжку мышц',
                'intensity': 'Низкая',
                'calories_per_hour': 150,
                'equipment': 'Коврик'
            }
        }
        
        # База тренировочных программ
        self.training_programs = {
            'weight_loss': [
                {
                    'id': 'wl_beginner',
                    'name': 'Похудение для начинающих',
                    'level': 'Начальный',
                    'description': 'Программа для мягкого начала похудения',
                    'duration_weeks': 8,
                    'sessions_per_week': 3,
                    'session_duration': 40,
                    'activities': ['cardio', 'circuit_training', 'pilates'],
                    'schedule': [
                        'День 1: Кардио 30 мин + Растяжка 10 мин',
                        'День 2: Круговая тренировка 40 мин',
                        'День 3: Пилатес 30 мин + Кардио 10 мин'
                    ],
                    'nutrition_tips': [
                        'Пейте 2 литра воды в день',
                        'Увеличьте потребление белка',
                        'Снизьте потребление быстрых углеводов'
                    ],
                    'progress_tracking': [
                        'Вес 1 раз в неделю',
                        'Объемы талии и бедер каждые 2 недели',
                        'Фото прогресса каждые 4 недели'
                    ]
                },
                {
                    'id': 'wl_intensive',
                    'name': 'Интенсивное похудение',
                    'level': 'Средний',
                    'description': 'Программа для быстрого снижения веса',
                    'duration_weeks': 6,
                    'sessions_per_week': 5,
                    'session_duration': 50,
                    'activities': ['circuit_training', 'cardio', 'strength'],
                    'schedule': [
                        'День 1: ВИИТ кардио 30 мин',
                        'День 2: Силовая тренировка 40 мин',
                        'День 3: Круговая тренировка 45 мин',
                        'День 4: Активный отдых (ходьба)',
                        'День 5: Интервальное кардио 35 мин'
                    ],
                    'nutrition_tips': [
                        'Дефицит калорий 300-500 ккал в день',
                        '5-6 небольших приемов пищи',
                        'Белок 1.5г на кг веса'
                    ]
                }
            ],
            'muscle_gain': [
                {
                    'id': 'mg_beginner',
                    'name': 'Набор массы для начинающих',
                    'level': 'Начальный',
                    'description': 'Базовая программа для набора мышечной массы',
                    'duration_weeks': 12,
                    'sessions_per_week': 4,
                    'session_duration': 60,
                    'activities': ['strength', 'cardio'],
                    'schedule': [
                        'День 1: Верх тела (грудь, спина)',
                        'День 2: Ноги',
                        'День 3: Отдых',
                        'День 4: Плечи, руки',
                        'День 5: Кардио 20 мин'
                    ]
                }
            ],
            'flexibility': [
                {
                    'id': 'flex_beginner',
                    'name': 'Йога для начинающих',
                    'level': 'Начальный',
                    'description': 'Программа для развития гибкости и расслабления',
                    'duration_weeks': 4,
                    'sessions_per_week': 5,
                    'session_duration': 30,
                    'activities': ['yoga', 'stretching'],
                    'schedule': [
                        'День 1: Утренняя йога 20 мин',
                        'День 2: Вечерняя растяжка 30 мин',
                        'День 3: Йога для спины 25 мин',
                        'День 4: Отдых',
                        'День 5: Полная сессия йоги 30 мин'
                    ]
                },
                {
                    'id': 'flex_pilates',
                    'name': 'Пилатес для гибкости',
                    'level': 'Средний',
                    'description': 'Программа пилатеса для развития гибкости',
                    'duration_weeks': 6,
                    'sessions_per_week': 4,
                    'session_duration': 45,
                    'activities': ['pilates', 'stretching'],
                    'schedule': [
                        'День 1: Пилатес для начинающих 40 мин',
                        'День 2: Растяжка 30 мин',
                        'День 3: Пилатес для пресса 45 мин',
                        'День 4: Йога-стретчинг 35 мин'
                    ]
                }
            ],
            'endurance': [
                {
                    'id': 'end_beginner',
                    'name': 'Кардио для выносливости',
                    'level': 'Начальный',
                    'description': 'Программа для улучшения кардио-выносливости',
                    'duration_weeks': 8,
                    'sessions_per_week': 3,
                    'session_duration': 40,
                    'activities': ['cardio', 'circuit_training'],
                    'schedule': [
                        'День 1: Бег/Ходьба 30 мин',
                        'День 2: Велотренажер 35 мин',
                        'День 3: Круговая тренировка 40 мин'
                    ]
                }
            ],
            'health': [
                {
                    'id': 'health_balance',
                    'name': 'Сбалансированное здоровье',
                    'level': 'Начальный',
                    'description': 'Программа для общего оздоровления',
                    'duration_weeks': 8,
                    'sessions_per_week': 4,
                    'session_duration': 40,
                    'activities': ['yoga', 'cardio', 'strength'],
                    'schedule': [
                        'День 1: Йога для расслабления 30 мин',
                        'День 2: Легкое кардио 30 мин',
                        'День 3: Силовая тренировка 40 мин',
                        'День 4: Активная прогулка 45 мин'
                    ],
                    'nutrition_tips': [
                        'Сбалансированное питание',
                        'Достаточное количество воды',
                        'Регулярные приемы пищи'
                    ]
                }
            ]
        }
    
    def init_ml_model(self):
        """Инициализация ML модели для подбора тренировок"""
        model_path = os.path.join(self.data_dir, 'training_recommender.pkl')
        
        if os.path.exists(model_path):
            try:
                # Загружаем существующую модель
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(os.path.join(self.data_dir, 'scaler.pkl'))
                return True
            except:
                # Если ошибка загрузки, создаем новую модель
                return self.train_recommendation_model()
        else:
            # Создаем и обучаем модель на синтетических данных
            return self.train_recommendation_model()
    
    def train_recommendation_model(self):
        """Обучает модель на синтетических данных"""
        try:
            # Создаем синтетические данные для обучения
            np.random.seed(42)
            n_samples = 1000
            
            # Признаки: возраст, вес, рост, пол (0-жен,1-муж)
            X = np.zeros((n_samples, 4))
            X[:, 0] = np.random.randint(18, 65, n_samples)  # возраст
            X[:, 1] = np.random.normal(70, 15, n_samples)   # вес
            X[:, 2] = np.random.normal(170, 10, n_samples)  # рост
            X[:, 3] = np.random.randint(0, 2, n_samples)    # пол
            
            # Рассчитываем ИМТ
            bmi = X[:, 1] / ((X[:, 2] / 100) ** 2)
            
            # Цели на основе ИМТ и других факторов
            y = []
            for i in range(n_samples):
                if bmi[i] > 25:
                    y.append('weight_loss')  # Похудение
                elif bmi[i] < 18.5:
                    y.append('muscle_gain')  # Набор массы
                elif X[i, 0] > 50:
                    y.append('flexibility')  # Гибкость для старшего возраста
                elif X[i, 0] < 30 and X[i, 3] == 1:  # Молодые мужчины
                    y.append('muscle_gain')
                else:
                    y.append('endurance')    # Выносливость
            
            # Кодируем цели
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Масштабируем признаки
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Обучаем модель
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y_encoded)
            
            # Сохраняем модель и скейлер
            joblib.dump(self.model, os.path.join(self.data_dir, 'training_recommender.pkl'))
            joblib.dump(self.scaler, os.path.join(self.data_dir, 'scaler.pkl'))
            joblib.dump(le, os.path.join(self.data_dir, 'label_encoder.pkl'))
            return True
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")
            return False
    
    def recommend_programs_based_on_profile(self, user_profile):
        """Рекомендует программы тренировок на основе профиля пользователя"""
        try:
            personal_info = user_profile.get('personal_info', {})
            goals = user_profile.get('goals', {})
            preferred_activities = user_profile.get('preferred_activities', [])
            
            # Извлекаем данные для ML модели
            age = personal_info.get('age', 30)
            weight = personal_info.get('weight', 70)
            height = personal_info.get('height', 170)
            gender = 0 if personal_info.get('gender') == 'Женский' else 1
            primary_goal = goals.get('primary_goal', 'weight_loss')
            
            # Подготавливаем признаки для модели (4 признака как при обучении)
            X = np.array([[age, weight, height, gender]])
            
            # Проверяем, есть ли модель
            if not hasattr(self, 'model') or self.model is None:
                # Возвращаем программы по выбранной цели
                final_goal = primary_goal if primary_goal in self.training_programs else 'weight_loss'
                return self.training_programs.get(final_goal, [])[:3]
            
            # Масштабируем признаки
            X_scaled = self.scaler.transform(X)
            
            # Предсказываем цель
            le_path = os.path.join(self.data_dir, 'label_encoder.pkl')
            if os.path.exists(le_path):
                le = joblib.load(le_path)
                predicted_goal_encoded = self.model.predict(X_scaled)[0]
                predicted_goal = le.inverse_transform([predicted_goal_encoded])[0]
            else:
                predicted_goal = primary_goal
            
            # Используем либо выбранную пользователем цель, либо предсказанную
            final_goal = primary_goal if primary_goal in self.goals else predicted_goal
            
            # Получаем программы для цели
            recommended_programs = self.training_programs.get(final_goal, [])
            
            # Фильтруем по предпочитаемым активностям
            if preferred_activities and recommended_programs:
                filtered_programs = []
                for program in recommended_programs:
                    program_activities = program.get('activities', [])
                    # Проверяем, есть ли пересечение с предпочитаемыми активностями
                    if any(activity in preferred_activities for activity in program_activities):
                        filtered_programs.append(program)
                
                if filtered_programs:
                    return filtered_programs[:3]  # Возвращаем до 3 программ
            
            return recommended_programs[:3]
            
        except Exception as e:
            # В случае ошибки возвращаем программы по умолчанию
            st.warning(f"Используются рекомендации по умолчанию")
            primary_goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
            return self.training_programs.get(primary_goal, self.training_programs['weight_loss'])[:3]
    
    def calculate_calories_needed(self, user_profile):
        """Рассчитывает суточную потребность в калориях"""
        personal_info = user_profile.get('personal_info', {})
        
        weight = personal_info.get('weight', 70)
        height = personal_info.get('height', 170)
        age = personal_info.get('age', 30)
        gender = personal_info.get('gender', 'Женский')
        activity_level = personal_info.get('activity_level', 'sedentary')
        
        # Базальный метаболизм (формула Миффлина-Сан Жеора)
        if gender == 'Мужской':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        # Коэффициент активности
        activity_multipliers = {
            'sedentary': 1.2,      # Сидячий образ жизни
            'light': 1.375,        # Легкая активность 1-3 раза в неделю
            'moderate': 1.55,      # Умеренная активность 3-5 раз в неделю
            'active': 1.725,       # Высокая активность 6-7 раз в неделю
            'very_active': 1.9,    # Очень высокая активность
        }
        
        tdee = bmr * activity_multipliers.get(activity_level, 1.2)
        
        # Корректировка по цели
        goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
        if goal == 'weight_loss':
            calories = tdee - 500  # Дефицит для похудения
        elif goal == 'muscle_gain':
            calories = tdee + 300  # Профицит для набора массы
        else:
            calories = tdee  # Поддержание
        
        return int(calories), int(tdee)
    
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
            
            # Создаем профиль
            profile = {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {},
                'preferred_activities': [],
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
            st.error(f"Ошибка сохранения профиля: {e}")
            return False
    
    def load_user_profile(self, username):
        """Загружает профиль пользователя"""
        try:
            filename = self.get_user_profile_filename(username)
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    profile = json.load(f)
                # Проверяем структуру профиля
                if 'questionnaire_completed' not in profile:
                    profile['questionnaire_completed'] = False
                if 'preferred_activities' not in profile:
                    profile['preferred_activities'] = []
                return profile
            else:
                return {
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'personal_info': {},
                    'goals': {},
                    'preferred_activities': [],
                    'questionnaire_completed': False
                }
        except:
            return {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {},
                'preferred_activities': [],
                'questionnaire_completed': False
            }
    
    def complete_questionnaire(self, username, personal_info, goals, preferred_activities):
        """Завершает анкету пользователя"""
        profile = self.load_user_profile(username)
        profile['personal_info'] = personal_info
        profile['goals'] = goals
        profile['preferred_activities'] = preferred_activities
        profile['questionnaire_completed'] = True
        profile['questionnaire_date'] = datetime.now().isoformat()
        
        # Рассчитываем ИМТ
        height_m = personal_info['height'] / 100
        bmi = personal_info['weight'] / (height_m ** 2)
        profile['bmi'] = round(bmi, 1)
        profile['bmi_category'] = self.get_bmi_category(bmi)
        
        return self.save_user_profile(username, profile)
    
    def get_bmi_category(self, bmi):
        """Определяет категорию ИМТ"""
        if bmi < 18.5:
            return 'Недостаточный вес'
        elif bmi < 25:
            return 'Нормальный вес'
        elif bmi < 30:
            return 'Избыточный вес'
        else:
            return 'Ожирение'
    
    def add_workout(self, username, workout_type, duration, intensity, notes=''):
        """Добавляет тренировку пользователя"""
        try:
            new_data = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'workout_type': workout_type,
                'duration': int(duration),
                'intensity': intensity,
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
    
    def get_all_workouts(self, username):
        """Возвращает все тренировки пользователя"""
        filename = self.get_user_filename(username)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date', ascending=False)
        else:
            return pd.DataFrame(columns=['date', 'workout_type', 'duration', 'intensity', 'notes'])
    
    def get_statistics(self, username):
        """Возвращает статистику тренировок пользователя"""
        df = self.get_all_workouts(username)
        if df.empty:
            return {}
        
        stats = {
            'total_workouts': len(df),
            'total_minutes': df['duration'].sum(),
            'avg_duration': df['duration'].mean(),
            'workouts_this_month': len(df[df['date'] >= (datetime.now() - timedelta(days=30))]),
            'last_workout': df['date'].max() if not df.empty else None,
            'workout_streak': self.calculate_streak(df),
            'favorite_workout': df['workout_type'].mode().iloc[0] if not df['workout_type'].mode().empty else "Нет данных"
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
    
    def get_achievements(self, username):
        """Возвращает достижения пользователя"""
        stats = self.get_statistics(username)
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
        
        if stats.get('total_workouts', 0) >= 30:
            achievements.append({
                'id': 'consistent',
                'title': '📅 Регулярность',
                'description': '30 выполненных тренировок',
                'icon': '📅',
                'unlocked': True
            })
        
        if stats.get('total_minutes', 0) >= 1000:
            achievements.append({
                'id': 'thousand_minutes',
                'title': '⏱️ 1000 минут',
                'description': '1000 минут тренировок',
                'icon': '⏱️',
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
        
        # Достижение за заполнение анкеты
        if profile.get('questionnaire_completed', False):
            achievements.append({
                'id': 'questionnaire_complete',
                'title': '📝 Анкета заполнена',
                'description': 'Вы заполнили свою анкету',
                'icon': '📝',
                'unlocked': True
            })
        
        # Достижение за прогресс по весу
        if profile.get('personal_info', {}).get('weight') and profile.get('goals', {}).get('target_weight'):
            current = profile['personal_info']['weight']
            target = profile['goals']['target_weight']
            if abs(current - target) <= 2:  # Достигли цели в пределах 2 кг
                achievements.append({
                    'id': 'goal_achieved',
                    'title': '🏆 Цель достигнута!',
                    'description': f'Достигнут целевой вес {target}кг',
                    'icon': '🏆',
                    'unlocked': True
                })
        
        return achievements

# Инициализация приложения
app = FitnessAssistant()

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
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Главная"

initialize_session_state()

# Страница входа/регистрации
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">🧘 Фитнес Помощник</h1>', unsafe_allow_html=True)
    
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
                register_clicked = st.form_submit_button("Регистрация", use_container_width=True)
                if register_clicked:
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
                back_clicked = st.form_submit_button("Назад к входу", use_container_width=True)
                if back_clicked:
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
            age = st.number_input("Возраст:", min_value=10, max_value=100, value=25, key="q_age")
            height = st.number_input("Рост (см):", min_value=100, max_value=250, value=170, key="q_height")
        with col2:
            weight = st.number_input("Текущий вес (кг):", min_value=30, max_value=200, value=70, key="q_weight")
            gender = st.selectbox("Пол:", ["Женский", "Мужской"], key="q_gender")
        
        st.subheader("🎯 Ваши цели")
        
        primary_goal = st.selectbox("Основная цель:", 
                                  ["Похудение", "Набор мышечной массы", "Улучшение выносливости", 
                                   "Развитие гибкости", "Общее оздоровление"], key="q_goal")
        
        target_weight = st.number_input("Желаемый вес (кг):", min_value=30, max_value=200, value=65, key="q_target_weight")
        
        st.subheader("🏋️‍♀️ Предпочитаемые виды активности")
        st.write("Выберите виды тренировок, которые вам нравятся:")
        
        # Мультивыбор активностей
        activity_options = list(app.activity_types.keys())
        activity_names = [app.activity_types[a]['name'] for a in activity_options]
        
        selected_indices = st.multiselect(
            "Выберите предпочитаемые активности:",
            options=range(len(activity_names)),
            format_func=lambda x: f"{app.activity_types[activity_options[x]]['icon']} {activity_names[x]}",
            default=[0, 1, 2],  # По умолчанию йога, пилатес, круговые
            key="q_activities"
        )
        
        preferred_activities = [activity_options[i] for i in selected_indices]
        
        st.subheader("📊 Уровень активности")
        activity_level = st.select_slider(
            "Как часто вы тренируетесь?",
            options=["Сидячий", "Легкая активность", "Умеренная", "Высокая", "Очень высокая"],
            value="Умеренная",
            key="q_activity_level"
        )
        
        level_mapping = {
            "Сидячий": "sedentary",
            "Легкая активность": "light",
            "Умеренная": "moderate",
            "Высокая": "active",
            "Очень высокая": "very_active"
        }
        
        submitted = st.form_submit_button("✅ Сохранить анкету", use_container_width=True)
        
        if submitted:
            personal_info = {
                'age': age,
                'height': height,
                'weight': weight,
                'gender': gender,
                'activity_level': level_mapping[activity_level]
            }
            
            goals = {
                'primary_goal': {
                    'Похудение': 'weight_loss',
                    'Набор мышечной массы': 'muscle_gain',
                    'Улучшение выносливости': 'endurance',
                    'Развитие гибкости': 'flexibility',
                    'Общее оздоровление': 'health'
                }[primary_goal],
                'target_weight': target_weight
            }
            
            if app.complete_questionnaire(st.session_state.current_user, personal_info, goals, preferred_activities):
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
    st.sidebar.markdown(f'<div class="user-card">👤 {st.session_state.current_user}</div>', unsafe_allow_html=True)
    
    # Показываем цель пользователя
    if user_profile.get('goals', {}).get('primary_goal'):
        goal_info = app.goals.get(user_profile['goals']['primary_goal'], {})
        if goal_info:
            st.sidebar.markdown(f"""
            <div style='text-align: center; margin: 1rem 0;'>
                <span class='sport-icon'>{goal_info['icon']}</span>
                <h4>{goal_info['name']}</h4>
                <span class='goal-badge {goal_info["color"]}'>{goal_info['description']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Основная навигация
    with st.sidebar:
        st.title("Навигация")
        
        page = st.radio(
            "Выберите раздел:",
            ["📊 Главная", "🎯 Мои программы", "➕ Добавить тренировку", "📈 Мой прогресс", "🏆 Достижения", "👤 Мой профиль"],
            index=["📊 Главная", "🎯 Мои программы", "➕ Добавить тренировку", "📈 Мой прогресс", "🏆 Достижения", "👤 Мой профиль"].index(st.session_state.current_page)
        )
        
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Статистика")
        
        stats = app.get_statistics(st.session_state.current_user)
        if stats:
            st.metric("Всего тренировок", stats['total_workouts'])
            st.metric("Общее время", f"{int(stats['total_minutes'])} мин")
            if stats.get('workout_streak', 0) > 0:
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
        
        if not user_profile.get('questionnaire_completed', False):
            st.warning("""
            ⚠️ **Анкета не заполнена!**
            
            Для получения персонализированных рекомендаций заполните анкету.
            """)
            if st.button("📝 Заполнить анкету", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
        else:
            # Персональная информация
            personal_info = user_profile.get('personal_info', {})
            goals = user_profile.get('goals', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                st.metric("Текущий вес", f"{personal_info.get('weight', 0)} кг")
                st.metric("Целевой вес", f"{goals.get('target_weight', 0)} кг")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                bmi = user_profile.get('bmi', 0)
                bmi_category = user_profile.get('bmi_category', '')
                st.metric("ИМТ", f"{bmi}")
                st.caption(f"Категория: {bmi_category}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                calories_needed, tdee = app.calculate_calories_needed(user_profile)
                st.metric("Калории в день", f"{calories_needed}")
                st.caption(f"Расход: {tdee} ккал")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Рекомендуемые программы на основе ML
            st.markdown("### 🎯 Персональные рекомендации")
            
            recommended_programs = app.recommend_programs_based_on_profile(user_profile)
            
            if recommended_programs:
                for program in recommended_programs:
                    with st.container():
                        # Получаем информацию об активностях
                        activity_icons = ""
                        for activity_id in program.get('activities', []):
                            activity = app.activity_types.get(activity_id, {})
                            activity_icons += f"{activity.get('icon', '🏃')} "
                        
                        st.markdown(f"""
                        <div class="training-card">
                            <h3>{activity_icons} {program['name']}</h3>
                            <p><strong>Уровень:</strong> {program['level']} | <strong>Продолжительность:</strong> {program['duration_weeks']} недель</p>
                            <p>{program['description']}</p>
                            <p><strong>Расписание:</strong></p>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        for session in program.get('schedule', []):
                            st.markdown(f"<li>{session}</li>", unsafe_allow_html=True)
                        
                        st.markdown("</ul>", unsafe_allow_html=True)
                        
                        # Советы по питанию
                        if 'nutrition_tips' in program:
                            st.markdown("<p><strong>Советы по питанию:</strong></p><ul>", unsafe_allow_html=True)
                            for tip in program['nutrition_tips']:
                                st.markdown(f"<li>{tip}</li>", unsafe_allow_html=True)
                            st.markdown("</ul>", unsafe_allow_html=True)
                        
                        # Кнопка для выбора программы
                        if st.button(f"🎯 Выбрать эту программу", key=f"select_{program['id']}", use_container_width=True):
                            st.success(f"✅ Программа '{program['name']}' выбрана!")
            else:
                st.info("""
                💡 **Рекомендации появятся после заполнения анкеты.**
                
                Наш ИИ анализирует ваши данные и подбирает оптимальные тренировочные программы.
                """)
            
            # Быстрые действия
            st.markdown("### ⚡ Быстрые действия")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("➕ Добавить тренировку", use_container_width=True):
                    st.session_state.current_page = "➕ Добавить тренировку"
                    st.rerun()
            with col2:
                if st.button("📈 Мой прогресс", use_container_width=True):
                    st.session_state.current_page = "📈 Мой прогресс"
                    st.rerun()
            with col3:
                if st.button("🏆 Мои достижения", use_container_width=True):
                    st.session_state.current_page = "🏆 Достижения"
                    st.rerun()

    # Мои программы
    elif st.session_state.current_page == "🎯 Мои программы":
        st.markdown('<h2 class="sub-header">🎯 Мои тренировочные программы</h2>', unsafe_allow_html=True)
        
        if not user_profile.get('questionnaire_completed', False):
            st.warning("Заполните анкету для получения персональных программ")
        else:
            # Показываем все программы для цели пользователя
            goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
            goal_programs = app.training_programs.get(goal, [])
            
            if goal_programs:
                st.success(f"📊 Найдено {len(goal_programs)} программ для вашей цели")
                
                for program in goal_programs:
                    with st.expander(f"{program['name']} ({program['level']})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Описание:** {program['description']}")
                            st.write(f"**Продолжительность:** {program['duration_weeks']} недель")
                            st.write(f"**Тренировок в неделю:** {program['sessions_per_week']}")
                            st.write(f"**Длительность тренировки:** {program['session_duration']} минут")
                            
                            st.write("**Расписание:**")
                            for session in program.get('schedule', []):
                                st.write(f"- {session}")
                        
                        with col2:
                            # Показываем иконки активностей
                            st.write("**Активности:**")
                            for activity_id in program.get('activities', []):
                                activity = app.activity_types.get(activity_id, {})
                                st.write(f"{activity.get('icon', '🏃')} {activity.get('name', activity_id)}")
                            
                            if st.button(f"✅ Начать программу", key=f"start_{program['id']}"):
                                st.success(f"Программа '{program['name']}' начата!")
            else:
                st.info("Программы для вашей цели находятся в разработке. Скоро появятся!")

    # Добавление тренировки
    elif st.session_state.current_page == "➕ Добавить тренировку":
        st.markdown('<h2 class="sub-header">➕ Добавить тренировку</h2>', unsafe_allow_html=True)
        
        with st.form("add_workout_form"):
            # Получаем доступные виды тренировок из предпочитаемых активностей
            preferred_activities = user_profile.get('preferred_activities', [])
            
            if preferred_activities:
                workout_options = []
                workout_mapping = {}
                for activity_id in preferred_activities:
                    activity = app.activity_types.get(activity_id, {})
                    display_name = f"{activity.get('icon', '🏃')} {activity.get('name', activity_id)}"
                    workout_options.append(display_name)
                    workout_mapping[display_name] = activity.get('name', activity_id)
                
                workout_type = st.selectbox(
                    "Вид тренировки:",
                    options=workout_options,
                    key="workout_type_select"
                )
                workout_type_clean = workout_mapping[workout_type]
            else:
                workout_type = st.text_input("Вид тренировки:", placeholder="Например: Йога, Бег, Пилатес...", key="workout_type_text")
                workout_type_clean = workout_type
            
            col1, col2 = st.columns(2)
            with col1:
                duration = st.number_input("Длительность (минут):", min_value=5, max_value=180, value=45, key="workout_duration")
            with col2:
                intensity = st.select_slider(
                    "Интенсивность:",
                    options=["Очень легкая", "Легкая", "Средняя", "Высокая", "Очень высокая"],
                    value="Средняя",
                    key="workout_intensity"
                )
            
            notes = st.text_area("Заметки:", placeholder="Как прошла тренировка? Что понравилось?", key="workout_notes")
            
            submit_button = st.form_submit_button("💾 Сохранить тренировку", use_container_width=True)
            
            if submit_button:
                success, message = app.add_workout(
                    st.session_state.current_user, 
                    workout_type_clean, 
                    duration, 
                    intensity, 
                    notes
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                    st.rerun()
                else:
                    st.error(message)

    # Мой прогресс
    elif st.session_state.current_page == "📈 Мой прогресс":
        st.markdown('<h2 class="sub-header">📈 Мой прогресс</h2>', unsafe_allow_html=True)
        
        # Статистика тренировок
        stats = app.get_statistics(st.session_state.current_user)
        workouts = app.get_all_workouts(st.session_state.current_user)
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего тренировок", stats['total_workouts'])
            with col2:
                st.metric("Общее время", f"{int(stats['total_minutes'])} мин")
            with col3:
                if not pd.isna(stats['avg_duration']):
                    st.metric("Средняя длительность", f"{stats['avg_duration']:.0f} мин")
                else:
                    st.metric("Средняя длительность", "0 мин")
            with col4:
                st.metric("Текущая серия", f"{stats.get('workout_streak', 0)} дней")
        
        # График тренировок
        if not workouts.empty:
            st.markdown("### 📊 График активности")
            
            # Группируем по дням
            workouts['date_only'] = workouts['date'].dt.date
            daily_workouts = workouts.groupby('date_only').agg({
                'duration': 'sum',
                'workout_type': 'count'
            }).reset_index()
            daily_workouts.columns = ['date', 'total_minutes', 'workout_count']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # График 1: Длительность тренировок по дням
            ax1.bar(daily_workouts['date'], daily_workouts['total_minutes'], color='#4CAF50')
            ax1.set_title('Длительность тренировок по дням', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Минуты')
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # График 2: Количество тренировок по дням
            ax2.bar(daily_workouts['date'], daily_workouts['workout_count'], color='#2196F3')
            ax2.set_title('Количество тренировок по дням', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Тренировки')
            ax2.set_xlabel('Дата')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Таблица последних тренировок
            st.markdown("### 📋 История тренировок")
            recent_workouts = workouts.head(10).copy()
            recent_workouts['date'] = recent_workouts['date'].dt.strftime('%d.%m.%Y %H:%M')
            st.dataframe(recent_workouts[['date', 'workout_type', 'duration', 'intensity', 'notes']], 
                        use_container_width=True, hide_index=True)
        else:
            st.info("📝 У вас пока нет тренировок. Добавьте первую!")

    # Достижения
    elif st.session_state.current_page == "🏆 Достижения":
        st.markdown('<h2 class="sub-header">🏆 Мои достижения</h2>', unsafe_allow_html=True)
        
        achievements = app.get_achievements(st.session_state.current_user)
        stats = app.get_statistics(st.session_state.current_user)
        
        if achievements:
            unlocked = [a for a in achievements if a.get('unlocked', False)]
            total = len(achievements)
            
            st.success(f"🎉 У вас {len(unlocked)} из {total} достижений!")
            
            # Прогресс-бар
            if total > 0:
                progress = len(unlocked) / total * 100
                st.progress(min(int(progress), 100) / 100)
                st.caption(f"Прогресс: {len(unlocked)}/{total} ({progress:.1f}%)")
            
            # Отображение достижений
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
            
            # Ближайшие цели
            if stats:
                st.markdown("### 🎯 Ближайшие цели")
                goals_data = []
                
                if stats.get('total_workouts', 0) < 10:
                    goals_data.append(["🔥 Посвящение", f"{stats['total_workouts']}/10", "10 тренировок"])
                elif stats.get('total_workouts', 0) < 30:
                    goals_data.append(["📅 Регулярность", f"{stats['total_workouts']}/30", "30 тренировок"])
                
                if stats.get('total_minutes', 0) < 1000:
                    goals_data.append(["⏱️ 1000 минут", f"{int(stats['total_minutes'])}/1000", "1000 минут тренировок"])
                
                if stats.get('workout_streak', 0) < 7:
                    goals_data.append(["📆 Недельная серия", f"{stats['workout_streak']}/7", "7 дней подряд"])
                
                if goals_data:
                    goals_df = pd.DataFrame(goals_data, columns=['Достижение', 'Прогресс', 'Осталось'])
                    st.dataframe(goals_df, use_container_width=True, hide_index=True)
                else:
                    st.success("🎊 Все основные цели достигнуты! Вы настоящий чемпион! 🏆")
        else:
            st.info("""
            **Начните тренироваться чтобы получить достижения!** 🏋️‍♀️
            
            **Доступные достижения:**
            🎖️ **Первая тренировка** - Выполните первую тренировку
            🔥 **Посвящение** - 10 тренировок
            📅 **Регулярность** - 30 тренировок
            ⏱️ **1000 минут** - 1000 минут тренировок
            📆 **Недельная серия** - 7 тренировок подряд
            🌟 **Месячная серия** - 30 тренировок подряд
            📝 **Анкета заполнена** - Заполнение анкеты
            🏆 **Цель достигнута** - Достижение целевого веса
            """)

    # Мой профиль
    elif st.session_state.current_page == "👤 Мой профиль":
        st.markdown('<h2 class="sub-header">👤 Мой профиль</h2>', unsafe_allow_html=True)
        
        if not user_profile.get('questionnaire_completed', False):
            st.warning("Анкета не заполнена. Заполните для получения персонализированных рекомендаций.")
            if st.button("📝 Заполнить анкету", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
        else:
            with st.form("update_profile_form"):
                st.subheader("📏 Личные данные")
                
                personal_info = user_profile.get('personal_info', {})
                goals = user_profile.get('goals', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Возраст:", min_value=10, max_value=100, 
                                         value=personal_info.get('age', 25), key="profile_age")
                    height = st.number_input("Рост (см):", min_value=100, max_value=250, 
                                           value=personal_info.get('height', 170), key="profile_height")
                with col2:
                    weight = st.number_input("Текущий вес (кг):", min_value=30, max_value=200, 
                                           value=personal_info.get('weight', 70), key="profile_weight")
                    gender = st.selectbox("Пол:", ["Женский", "Мужской"], 
                                         index=0 if personal_info.get('gender') == 'Женский' else 1, 
                                         key="profile_gender")
                
                st.subheader("🎯 Цели")
                
                # Получаем текущую цель
                current_goal_key = goals.get('primary_goal', 'weight_loss')
                
                # Создаем список для selectbox
                goal_options = list(app.goals.keys())
                goal_display_names = []
                for key in goal_options:
                    goal_info = app.goals[key]
                    goal_display_names.append(f"{goal_info['icon']} {goal_info['name']}")
                
                # Находим индекс текущей цели
                current_index = goal_options.index(current_goal_key) if current_goal_key in goal_options else 0
                
                selected_goal_display = st.selectbox(
                    "Основная цель:",
                    options=goal_display_names,
                    index=current_index,
                    key="profile_goal"
                )
                
                # Получаем ключ цели из выбранного отображаемого имени
                selected_goal_index = goal_display_names.index(selected_goal_display)
                primary_goal_key = goal_options[selected_goal_index]
                
                target_weight = st.number_input("Желаемый вес (кг):", min_value=30, max_value=200, 
                                              value=goals.get('target_weight', 65), key="profile_target_weight")
                
                st.subheader("🏋️‍♀️ Предпочитаемые активности")
                activity_options = list(app.activity_types.keys())
                activity_names = [app.activity_types[a]['name'] for a in activity_options]
                
                current_indices = []
                for activity_id in user_profile.get('preferred_activities', []):
                    if activity_id in activity_options:
                        current_indices.append(activity_options.index(activity_id))
                
                selected_indices = st.multiselect(
                    "Выберите предпочитаемые активности:",
                    options=range(len(activity_names)),
                    format_func=lambda x: f"{app.activity_types[activity_options[x]]['icon']} {activity_names[x]}",
                    default=current_indices,
                    key="profile_activities"
                )
                
                preferred_activities = [activity_options[i] for i in selected_indices]
                
                # Кнопка отправки формы ДОЛЖНА быть внутри формы
                submit_button = st.form_submit_button("💾 Обновить профиль", use_container_width=True)
                
                if submit_button:
                    personal_info = {
                        'age': age,
                        'height': height,
                        'weight': weight,
                        'gender': gender,
                        'activity_level': personal_info.get('activity_level', 'moderate')
                    }
                    
                    goals = {
                        'primary_goal': primary_goal_key,
                        'target_weight': target_weight
                    }
                    
                    if app.complete_questionnaire(st.session_state.current_user, personal_info, goals, preferred_activities):
                        st.success("✅ Профиль успешно обновлен!")
                        st.rerun()
                    else:
                        st.error("❌ Ошибка обновления профиля")

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧘 <strong>Фитнес Помощник v6.0</strong> | Умный подбор тренировок на основе ваших данных</p>
    <p>Ваш персональный тренер для любого вида фитнеса</p>
</div>
""", unsafe_allow_html=True)
