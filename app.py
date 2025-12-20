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
import warnings
warnings.filterwarnings('ignore')

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
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .progress-metric {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .progress-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
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
    .level-beginner { background: #4CAF50; color: white; }
    .level-intermediate { background: #2196F3; color: white; }
    .level-advanced { background: #FF9800; color: white; }
    .level-pro { background: #f44336; color: white; }
    .exercise-item {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .video-link {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        margin: 0.5rem 0;
    }
    .video-link:hover {
        background: #ff5252;
    }
    .feedback-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .retrain-notification {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    .feedback-button {
        font-size: 1.5rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: 2px solid #ddd;
        background: white;
        cursor: pointer;
        transition: all 0.3s;
    }
    .feedback-button:hover {
        background: #f0f0f0;
        transform: scale(1.1);
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class SelfLearningFitnessAssistant:
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
        
        # Уровни сложности
        self.levels = {
            'beginner': {
                'name': 'Начальный',
                'color': 'level-beginner',
                'description': 'Для новичков, без опыта тренировок'
            },
            'intermediate': {
                'name': 'Средний',
                'color': 'level-intermediate',
                'description': 'Для тех, кто занимается 3-6 месяцев'
            },
            'advanced': {
                'name': 'Продвинутый',
                'color': 'level-advanced',
                'description': 'Для опытных, занимающихся более 6 месяцев'
            },
            'pro': {
                'name': 'Профи',
                'color': 'level-pro',
                'description': 'Для профессионалов и спортсменов'
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
        
        # База тренировочных программ с конкретными тренировками и видео
        self.training_programs = {
            'weight_loss': [
                {
                    'id': 'wl_beginner',
                    'name': 'Похудение для начинающих',
                    'level': 'beginner',
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
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Кардио + Растяжка',
                            'warmup': '5 минут легкой ходьбы на месте',
                            'exercises': [
                                {'type': 'cardio', 'name': 'Бег на месте', 'duration': '10 минут'},
                                {'type': 'cardio', 'name': 'Прыжки со скакалкой', 'duration': '10 минут'},
                                {'type': 'cardio', 'name': 'Высокие колени', 'duration': '5 минут'},
                                {'type': 'stretching', 'name': 'Растяжка ног', 'duration': '5 минут'},
                                {'type': 'stretching', 'name': 'Растяжка спины', 'duration': '5 минут'}
                            ],
                            'cooldown': '5 минут глубокого дыхания',
                            'video_url': 'https://www.youtube.com/watch?v=dF4WvM1lC90',
                            'video_description': 'Полная тренировка для начинающих: кардио + растяжка'
                        },
                        'day2': {
                            'title': 'Круговая тренировка',
                            'warmup': '5 минут динамической растяжки',
                            'exercises': [
                                {'type': 'strength', 'name': 'Приседания без веса', 'sets': '3', 'reps': '15', 'rest': '30 сек'},
                                {'type': 'strength', 'name': 'Отжимания от колен', 'sets': '3', 'reps': '10', 'rest': '30 сек'},
                                {'type': 'strength', 'name': 'Планка', 'sets': '3', 'duration': '30 сек', 'rest': '30 сек'},
                                {'type': 'cardio', 'name': 'Бег на месте', 'sets': '3', 'duration': '1 мин', 'rest': '30 сек'},
                                {'type': 'strength', 'name': 'Выпады', 'sets': '3', 'reps': '12 на каждую ногу', 'rest': '30 сек'}
                            ],
                            'cooldown': '5 минут растяжки',
                            'video_url': 'https://www.youtube.com/watch?v=J7hZ1G7Qn3Q',
                            'video_description': 'Круговая тренировка для похудения'
                        },
                        'day3': {
                            'title': 'Пилатес + Кардио',
                            'warmup': '5 минут разминки',
                            'exercises': [
                                {'type': 'pilates', 'name': 'Сотня', 'sets': '3', 'duration': '1 мин', 'rest': '30 сек'},
                                {'type': 'pilates', 'name': 'Ролл-ап', 'sets': '3', 'reps': '10', 'rest': '30 сек'},
                                {'type': 'pilates', 'name': 'Плавание', 'sets': '3', 'duration': '1 мин', 'rest': '30 сек'},
                                {'type': 'cardio', 'name': 'Велосипед', 'sets': '3', 'duration': '5 мин', 'rest': '1 мин'}
                            ],
                            'cooldown': '5 минут растяжки',
                            'video_url': 'https://www.youtube.com/watch?v=JDcdhTuycOI',
                            'video_description': 'Пилатес для начинающих'
                        }
                    }
                },
                {
                    'id': 'wl_intermediate',
                    'name': 'Интенсивное похудение',
                    'level': 'intermediate',
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
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'ВИИТ кардио',
                            'warmup': '5 минут легкого бега',
                            'exercises': [
                                {'type': 'cardio', 'name': 'Спринт', 'sets': '10', 'duration': '30 сек', 'rest': '30 сек'},
                                {'type': 'cardio', 'name': 'Берпи', 'sets': '5', 'reps': '10', 'rest': '45 сек'}
                            ],
                            'cooldown': '5 минут ходьбы',
                            'video_url': 'https://www.youtube.com/watch?v=M0uO8X3_tEA',
                            'video_description': 'ВИИТ тренировка для сжигания жира'
                        }
                    }
                }
            ],
            'muscle_gain': [
                {
                    'id': 'mg_beginner',
                    'name': 'Набор массы для начинающих',
                    'level': 'beginner',
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
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Верх тела (грудь, спина)',
                            'warmup': '10 минут разминки',
                            'exercises': [
                                {'type': 'strength', 'name': 'Жим лежа', 'sets': '3', 'reps': '10-12', 'rest': '60 сек'},
                                {'type': 'strength', 'name': 'Тяга в наклоне', 'sets': '3', 'reps': '10-12', 'rest': '60 сек'},
                                {'type': 'strength', 'name': 'Отжимания', 'sets': '3', 'reps': 'макс', 'rest': '60 сек'}
                            ],
                            'cooldown': '5 минут растяжки',
                            'video_url': 'https://www.youtube.com/watch?v=9efgcAjQe7E',
                            'video_description': 'Тренировка верхней части тела'
                        }
                    }
                }
            ],
            'flexibility': [
                {
                    'id': 'flex_beginner',
                    'name': 'Йога для начинающих',
                    'level': 'beginner',
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
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Утренняя йога',
                            'warmup': '5 минут дыхательных упражнений',
                            'exercises': [
                                {'type': 'yoga', 'name': 'Поза горы', 'duration': '2 минуты'},
                                {'type': 'yoga', 'name': 'Поза ребенка', 'duration': '3 минуты'},
                                {'type': 'yoga', 'name': 'Поза кошки-коровы', 'duration': '5 минут'}
                            ],
                            'cooldown': '5 минут медитации',
                            'video_url': 'https://www.youtube.com/watch?v=VaoV1PrYft4',
                            'video_description': 'Утренняя йога для начинающих'
                        }
                    }
                }
            ],
            'endurance': [
                {
                    'id': 'end_beginner',
                    'name': 'Кардио для выносливости',
                    'level': 'beginner',
                    'description': 'Программа для улучшения кардио-выносливости',
                    'duration_weeks': 8,
                    'sessions_per_week': 3,
                    'session_duration': 40,
                    'activities': ['cardio', 'circuit_training'],
                    'schedule': [
                        'День 1: Бег/Ходьба 30 мин',
                        'День 2: Велотренажер 35 мин',
                        'День 3: Круговая тренировка 40 мин'
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Интервальная ходьба/бег',
                            'warmup': '5 минут быстрой ходьбы',
                            'exercises': [
                                {'type': 'cardio', 'name': 'Ходьба', 'duration': '5 минут'},
                                {'type': 'cardio', 'name': 'Легкий бег', 'duration': '1 минута'},
                                {'type': 'cardio', 'name': 'Ходьба', 'duration': '2 минуты'},
                                {'type': 'cardio', 'name': 'Легкий бег', 'duration': '1 минута'},
                                {'type': 'cardio', 'name': 'Ходьба', 'duration': '5 минут'}
                            ],
                            'cooldown': '5 минут медленной ходьбы',
                            'video_url': 'https://www.youtube.com/watch?v=J7hZ1G7Qn3Q',
                            'video_description': 'Интервальная тренировка для начинающих'
                        }
                    }
                }
            ],
            'health': [
                {
                    'id': 'health_beginner',
                    'name': 'Сбалансированное здоровье',
                    'level': 'beginner',
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
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Йога для расслабления',
                            'warmup': '5 минут дыхания',
                            'exercises': [
                                {'type': 'yoga', 'name': 'Поза горы', 'duration': '3 минуты'},
                                {'type': 'yoga', 'name': 'Поза ребенка', 'duration': '5 минуты'},
                                {'type': 'yoga', 'name': 'Наклон вперед', 'duration': '2 минуты'}
                            ],
                            'cooldown': '5 минут медитации',
                            'video_url': 'https://www.youtube.com/watch?v=4pKly2JojMw',
                            'video_description': 'Йога для релаксации'
                        }
                    }
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
                self.label_encoder = joblib.load(os.path.join(self.data_dir, 'label_encoder.pkl'))
                return True
            except Exception as e:
                # Если ошибка загрузки, создаем новую модель
                return self.train_initial_model()
        else:
            # Создаем и обучаем модель на синтетических данных
            return self.train_initial_model()
    
    def train_initial_model(self):
        """Обучает начальную модель на синтетических данных"""
        try:
            # Создаем более реалистичные синтетические данные
            np.random.seed(42)
            n_samples = 2000
            
            # Признаки: возраст, вес, рост, пол (0-жен,1-муж)
            X = np.zeros((n_samples, 5))  # Теперь 5 признаков включая ИМТ
            
            # Генерация реалистичных данных
            X[:, 0] = np.random.randint(16, 70, n_samples)  # возраст
            X[:, 1] = np.random.normal(75, 20, n_samples)   # вес
            X[:, 1] = np.clip(X[:, 1], 40, 150)  # Ограничиваем вес
            X[:, 2] = np.random.normal(170, 10, n_samples)  # рост
            X[:, 2] = np.clip(X[:, 2], 150, 210)  # Ограничиваем рост
            X[:, 3] = np.random.randint(0, 2, n_samples)    # пол
            
            # Рассчитываем ИМТ и добавляем как признак
            height_m = X[:, 2] / 100
            X[:, 4] = X[:, 1] / (height_m ** 2)  # ИМТ
            
            # Цели на основе ИМТ и других факторов (более сложная логика)
            y = []
            for i in range(n_samples):
                age = X[i, 0]
                bmi = X[i, 4]
                gender = X[i, 3]
                
                if bmi > 28:  # Ожирение
                    if age > 50:
                        y.append('health')
                    else:
                        y.append('weight_loss')
                elif bmi < 18.5:  # Недостаточный вес
                    if gender == 1:  # Мужчины
                        y.append('muscle_gain')
                    else:
                        y.append('health')
                elif age > 55:  # Пожилые
                    y.append('flexibility')
                elif age < 25 and gender == 1:  # Молодые мужчины
                    y.append('muscle_gain')
                elif bmi > 24 and bmi <= 28:  # Избыточный вес
                    y.append('weight_loss')
                else:
                    # Случайный выбор между выносливостью и здоровьем
                    y.append(np.random.choice(['endurance', 'health']))
            
            # Кодируем цели
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Масштабируем признаки
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Обучаем модель с настройками для инкрементального обучения
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                warm_start=True,  # Для инкрементального обучения
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y_encoded)
            
            # Сохраняем модель и скейлер
            joblib.dump(self.model, os.path.join(self.data_dir, 'training_recommender.pkl'))
            joblib.dump(self.scaler, os.path.join(self.data_dir, 'scaler.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.data_dir, 'label_encoder.pkl'))
            
            # Сохраняем информацию о начальной модели
            model_info = {
                'initial_training_date': datetime.now().isoformat(),
                'initial_samples': n_samples,
                'feature_names': ['age', 'weight', 'height', 'gender', 'bmi'],
                'classes': list(self.label_encoder.classes_)
            }
            with open(os.path.join(self.data_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"❌ Ошибка при обучении начальной модели: {e}")
            return False
    
    def collect_feedback(self, username, program_id, rating, user_goal, actual_goal=None, comment=''):
        """Собирает обратную связь от пользователя по рекомендации"""
        try:
            feedback_file = os.path.join(self.data_dir, 'user_feedback.csv')
            
            # Загружаем профиль пользователя для получения его данных
            profile = self.load_user_profile(username)
            personal_info = profile.get('personal_info', {})
            
            # Рассчитываем ИМТ если есть данные
            bmi = None
            if 'weight' in personal_info and 'height' in personal_info:
                height_m = personal_info['height'] / 100
                bmi = personal_info['weight'] / (height_m ** 2)
            
            # Создаем запись обратной связи
            feedback_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': hashlib.md5(username.encode()).hexdigest()[:8],  # Анонимный ID
                'user_age': personal_info.get('age'),
                'user_weight': personal_info.get('weight'),
                'user_height': personal_info.get('height'),
                'user_gender': 0 if personal_info.get('gender') == 'Женский' else 1,
                'user_bmi': bmi,
                'program_id': program_id,
                'recommended_goal': user_goal,
                'actual_user_goal': actual_goal if actual_goal else user_goal,
                'user_rating': rating,  # 1-5 звезд
                'user_comment': comment
            }
            
            # Сохраняем в CSV
            df = pd.DataFrame([feedback_data])
            if os.path.exists(feedback_file):
                existing_df = pd.read_csv(feedback_file)
                # Проверяем, не дублируется ли отзыв
                recent_feedback = existing_df[
                    (existing_df['user_id'] == feedback_data['user_id']) & 
                    (existing_df['program_id'] == program_id) &
                    (pd.to_datetime(existing_df['timestamp']) > pd.Timestamp.now() - pd.Timedelta(hours=1))
                ]
                if len(recent_feedback) == 0:
                    df.to_csv(feedback_file, mode='a', header=False, index=False)
                else:
                    return True, "Вы уже оставляли отзыв по этой программе недавно."
            else:
                df.to_csv(feedback_file, index=False)
            
            # Проверяем, нужно ли запустить дообучение
            self._check_retraining_needed()
            
            return True, "Спасибо за ваш отзыв! Он поможет улучшить рекомендации."
        except Exception as e:
            return False, f"Ошибка сохранения отзыва: {e}"
    
    def _check_retraining_needed(self):
        """Проверяет, нужно ли запустить дообучение модели"""
        try:
            feedback_file = os.path.join(self.data_dir, 'user_feedback.csv')
            if not os.path.exists(feedback_file):
                return
            
            feedback_df = pd.read_csv(feedback_file)
            
            # Если накопилось достаточно новых отзывов (например, 30)
            if len(feedback_df) >= 30:
                # Проверяем, когда последний раз дообучали модель
                retrain_log_path = os.path.join(self.data_dir, 'retraining_log.json')
                if os.path.exists(retrain_log_path):
                    with open(retrain_log_path, 'r') as f:
                        log = json.load(f)
                    last_retrain = pd.Timestamp(log[-1]['retrain_date']) if log else pd.Timestamp.min
                else:
                    last_retrain = pd.Timestamp.min
                
                # Если прошло больше 3 дней с последнего дообучения
                if (pd.Timestamp.now() - last_retrain).days >= 3:
                    # Автоматически запускаем дообучение
                    success, message = self.retrain_model_with_feedback()
                    if success:
                        st.session_state.auto_retrain_message = message
        except Exception as e:
            pass
    
    def retrain_model_with_feedback(self, force_retrain=False):
        """Дообучает модель на основе накопленных отзывов пользователей"""
        try:
            feedback_file = os.path.join(self.data_dir, 'user_feedback.csv')
            if not os.path.exists(feedback_file):
                return False, "Файл с отзывами не найден."
            
            feedback_df = pd.read_csv(feedback_file)
            
            # Фильтруем валидные записи
            valid_feedback = feedback_df.dropna(subset=['user_age', 'user_weight', 'user_height', 'user_gender', 'user_rating'])
            
            if len(valid_feedback) < 20 and not force_retrain:
                return False, f"Недостаточно данных для дообучения (только {len(valid_feedback)} записей). Нужно минимум 20."
            
            # Разделяем на положительные и все отзывы
            positive_feedback = valid_feedback[valid_feedback['user_rating'] >= 4]
            all_feedback = valid_feedback
            
            # Используем все отзывы, но взвешиваем по рейтингу
            weights = all_feedback['user_rating'] / 5.0  # Веса от 0.2 до 1.0
            
            # Подготовка признаков (теперь 5 признаков)
            X_new = all_feedback[['user_age', 'user_weight', 'user_height', 'user_gender']].values
            
            # Добавляем ИМТ как признак
            heights_m = all_feedback['user_height'] / 100
            bmis = all_feedback['user_weight'] / (heights_m ** 2)
            X_new = np.hstack([X_new, bmis.values.reshape(-1, 1)])
            
            # Используем actual_user_goal если есть, иначе recommended_goal
            y_new = all_feedback['actual_user_goal'].fillna(all_feedback['recommended_goal']).values
            
            # Загружаем текущую модель если еще не загружена
            if not hasattr(self, 'model'):
                self.init_ml_model()
            
            # 1. Полное переобучение на всех данных (старых + новых)
            # Для этого нам нужны старые данные
            old_data_path = os.path.join(self.data_dir, 'training_data.npz')
            
            if os.path.exists(old_data_path):
                # Загружаем старые данные
                old_data = np.load(old_data_path)
                X_old = old_data['X']
                y_old = old_data['y']
                
                # Объединяем старые и новые данные
                X_combined = np.vstack([X_old, X_new])
                y_combined = np.concatenate([y_old, y_new])
                
                # Ограничиваем общий размер данных (максимум 5000 примеров)
                if len(X_combined) > 5000:
                    X_combined = X_combined[-5000:]
                    y_combined = y_combined[-5000:]
            else:
                # Только новые данные
                X_combined = X_new
                y_combined = y_new
            
            # Сохраняем объединенные данные
            np.savez(old_data_path, X=X_combined, y=y_combined)
            
            # Масштабируем данные
            X_scaled = self.scaler.fit_transform(X_combined)
            
            # Кодируем цели
            y_encoded = self.label_encoder.transform(y_combined)
            
            # Переобучаем модель с нуля
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                warm_start=True,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y_encoded)
            
            # Сохраняем обновленную модель
            joblib.dump(self.model, os.path.join(self.data_dir, 'training_recommender.pkl'))
            joblib.dump(self.scaler, os.path.join(self.data_dir, 'scaler.pkl'))
            joblib.dump(self.label_encoder, os.path.join(self.data_dir, 'label_encoder.pkl'))
            
            # Логируем событие дообучения
            log_entry = {
                'retrain_date': datetime.now().isoformat(),
                'samples_used': len(X_combined),
                'new_samples': len(X_new),
                'positive_feedback': len(positive_feedback),
                'total_feedback': len(feedback_df)
            }
            
            log_path = os.path.join(self.data_dir, 'retraining_log.json')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log = json.load(f)
            else:
                log = []
            
            log.append(log_entry)
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)
            
            return True, f"✅ Модель успешно дообучена! Использовано {len(X_combined)} примеров ({len(X_new)} новых)."
        
        except Exception as e:
            return False, f"❌ Ошибка при дообучении модели: {e}"
    
    def get_model_info(self):
        """Возвращает информацию о текущей модели"""
        info = {
            'has_model': hasattr(self, 'model') and self.model is not None,
            'model_type': type(self.model).__name__ if hasattr(self, 'model') else 'None',
            'feature_count': self.model.n_features_in_ if hasattr(self, 'model') and hasattr(self.model, 'n_features_in_') else 0,
            'classes': list(self.label_encoder.classes_) if hasattr(self, 'label_encoder') else []
        }
        
        # Читаем дополнительную информацию из файлов
        try:
            info_file = os.path.join(self.data_dir, 'model_info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                info.update(model_info)
        except:
            pass
        
        return info
    
    def recommend_programs_based_on_profile(self, user_profile, display_feedback=True):
        """Рекомендует программы тренировок на основе профиля пользователя с системой обратной связи"""
        try:
            personal_info = user_profile.get('personal_info', {})
            goals = user_profile.get('goals', {})
            preferred_activities = user_profile.get('preferred_activities', [])
            
            # Извлекаем данные для ML модели
            age = personal_info.get('age', 30)
            weight = personal_info.get('weight', 70)
            height = personal_info.get('height', 170)
            gender = 0 if personal_info.get('gender') == 'Женский' else 1
            
            # Рассчитываем ИМТ
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            
            # Подготавливаем признаки для модели (5 признаков)
            X = np.array([[age, weight, height, gender, bmi]])
            
            # Проверяем, есть ли модель
            if not hasattr(self, 'model') or self.model is None:
                st.warning("ML модель не загружена. Используются рекомендации по выбранной цели.")
                primary_goal = goals.get('primary_goal', 'weight_loss')
                final_goal = primary_goal if primary_goal in self.training_programs else 'weight_loss'
                recommended_programs = self.training_programs.get(final_goal, [])[:3]
            else:
                # Масштабируем признаки
                X_scaled = self.scaler.transform(X)
                
                # Предсказываем цель
                predicted_goal_encoded = self.model.predict(X_scaled)[0]
                predicted_goal = self.label_encoder.inverse_transform([predicted_goal_encoded])[0]
                
                # Определяем финальную цель (предпочтение пользователя или предсказание модели)
                primary_goal = goals.get('primary_goal', predicted_goal)
                
                # Если пользователь выбрал цель, используем её, иначе используем предсказание модели
                # Но также можем учитывать предсказание модели как рекомендацию
                final_goal = primary_goal
                
                # Получаем программы для цели
                recommended_programs = self.training_programs.get(final_goal, [])
                
                # Если программ для выбранной цели нет, используем предсказание модели
                if not recommended_programs:
                    final_goal = predicted_goal
                    recommended_programs = self.training_programs.get(final_goal, [])
                
                # Добавляем объяснение рекомендации
                if display_feedback:
                    goal_info = self.goals.get(predicted_goal, {})
                    with st.expander("🤖 Как ИИ сделал эту рекомендацию?", expanded=False):
                        st.write(f"**На основе ваших данных:**")
                        st.write(f"- Возраст: {age} лет")
                        st.write(f"- Рост: {height} см, Вес: {weight} кг")
                        st.write(f"- ИМТ: {bmi:.1f} ({self.get_bmi_category(bmi)})")
                        st.write(f"**Модель рекомендует:** {goal_info.get('name', predicted_goal)}")
                        st.write(f"**Ваш выбор:** {self.goals.get(primary_goal, {}).get('name', primary_goal)}")
            
            # Фильтруем по предпочитаемым активностям
            if preferred_activities and recommended_programs:
                filtered_programs = []
                for program in recommended_programs:
                    program_activities = program.get('activities', [])
                    # Проверяем, есть ли пересечение с предпочитаемыми активностями
                    if any(activity in preferred_activities for activity in program_activities):
                        filtered_programs.append(program)
                
                if filtered_programs:
                    recommended_programs = filtered_programs[:3]
            
            # Ограничиваем количество программ
            recommended_programs = recommended_programs[:3]
            
            return recommended_programs
            
        except Exception as e:
            # В случае ошибки возвращаем программы по умолчанию
            st.warning(f"Используются рекомендации по умолчанию. Ошибка: {str(e)[:100]}")
            primary_goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
            return self.training_programs.get(primary_goal, self.training_programs['weight_loss'])[:3]
    
    def get_exercises_for_program(self, program_id, day=None):
        """Возвращает упражнения для конкретной программы и дня"""
        # Находим программу
        program = None
        for goal_programs in self.training_programs.values():
            for p in goal_programs:
                if p['id'] == program_id:
                    program = p
                    break
            if program:
                break
        
        if not program:
            return {}
        
        # Если есть конкретные тренировки в программе
        if 'workouts' in program:
            if day and day in program['workouts']:
                return program['workouts'][day]
            elif program['workouts']:
                # Возвращаем первую тренировку, если день не указан
                first_day = list(program['workouts'].keys())[0]
                return program['workouts'][first_day]
        
        return {}
    
    def get_all_workout_days(self, program_id):
        """Возвращает все дни тренировок для программы"""
        program = None
        for goal_programs in self.training_programs.values():
            for p in goal_programs:
                if p['id'] == program_id:
                    program = p
                    break
            if program:
                break
        
        if not program or 'workouts' not in program:
            return []
        
        return list(program['workouts'].keys())
    
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
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9,
        }
        
        tdee = bmr * activity_multipliers.get(activity_level, 1.2)
        
        # Корректировка по цели
        goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
        if goal == 'weight_loss':
            calories = tdee - 500
        elif goal == 'muscle_gain':
            calories = tdee + 300
        else:
            calories = tdee
        
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
                'questionnaire_completed': False,
                'current_program': None,
                'program_start_date': None
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
                    profile = json.load(f)
                # Проверяем структуру профиля
                if 'questionnaire_completed' not in profile:
                    profile['questionnaire_completed'] = False
                if 'preferred_activities' not in profile:
                    profile['preferred_activities'] = []
                if 'current_program' not in profile:
                    profile['current_program'] = None
                if 'program_start_date' not in profile:
                    profile['program_start_date'] = None
                return profile
            else:
                return {
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'personal_info': {},
                    'goals': {},
                    'preferred_activities': [],
                    'questionnaire_completed': False,
                    'current_program': None,
                    'program_start_date': None
                }
        except:
            return {
                'username': username,
                'created_at': datetime.now().isoformat(),
                'personal_info': {},
                'goals': {},
                'preferred_activities': [],
                'questionnaire_completed': False,
                'current_program': None,
                'program_start_date': None
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
    
    def set_current_program(self, username, program_id):
        """Устанавливает текущую программу для пользователя"""
        profile = self.load_user_profile(username)
        profile['current_program'] = program_id
        profile['program_start_date'] = datetime.now().isoformat()
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
    
    def add_workout(self, username, workout_type, duration, intensity, notes='', program_id=None, day=None):
        """Добавляет тренировку пользователя"""
        try:
            new_data = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'workout_type': workout_type,
                'duration': int(duration),
                'intensity': intensity,
                'notes': notes,
                'program_id': program_id if program_id else '',
                'day': day if day else ''
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
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date', ascending=False)
        return pd.DataFrame(columns=['date', 'workout_type', 'duration', 'intensity', 'notes', 'program_id', 'day'])
    
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
            if abs(current - target) <= 2:
                achievements.append({
                    'id': 'goal_achieved',
                    'title': '🏆 Цель достигнута!',
                    'description': f'Достигнут целевой вес {target}кг',
                    'icon': '🏆',
                    'unlocked': True
                })
        
        # Достижение за выбор программы
        if profile.get('current_program'):
            achievements.append({
                'id': 'program_started',
                'title': '📋 Программа начата',
                'description': 'Вы начали тренировочную программу',
                'icon': '📋',
                'unlocked': True
            })
        
        # Достижение за обратную связь
        feedback_file = os.path.join(self.data_dir, 'user_feedback.csv')
        if os.path.exists(feedback_file):
            try:
                feedback_df = pd.read_csv(feedback_file)
                user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
                user_feedback_count = len(feedback_df[feedback_df['user_id'] == user_hash])
                
                if user_feedback_count >= 5:
                    achievements.append({
                        'id': 'feedback_pro',
                        'title': '💬 Эксперт по обратной связи',
                        'description': f'Оставил {user_feedback_count} отзывов',
                        'icon': '💬',
                        'unlocked': True
                    })
            except:
                pass
        
        return achievements

# Инициализация приложения
app = SelfLearningFitnessAssistant()

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
    if 'show_program_details' not in st.session_state:
        st.session_state.show_program_details = None
    if 'selected_day' not in st.session_state:
        st.session_state.selected_day = None
    if 'show_admin_panel' not in st.session_state:
        st.session_state.show_admin_panel = False
    if 'auto_retrain_message' not in st.session_state:
        st.session_state.auto_retrain_message = None
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = {}
    if 'rating_temp' not in st.session_state:
        st.session_state.rating_temp = {}

initialize_session_state()

# Страница входа/регистрации
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">💪 Фитнес Помощник</h1>', unsafe_allow_html=True)
    
    # Показываем уведомление об автоматическом дообучении если есть
    if st.session_state.get('auto_retrain_message'):
        st.markdown(f'<div class="retrain-notification">🔄 {st.session_state.auto_retrain_message}</div>', unsafe_allow_html=True)
        st.session_state.auto_retrain_message = None
    
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
            default=[0, 1, 2],
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
    
    # Показываем уведомление об автоматическом дообучении если есть
    if st.session_state.get('auto_retrain_message'):
        st.markdown(f'<div class="retrain-notification">🔄 {st.session_state.auto_retrain_message}</div>', unsafe_allow_html=True)
        st.session_state.auto_retrain_message = None
    
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
    
    # Показываем текущую программу
    if user_profile.get('current_program'):
        current_program_id = user_profile['current_program']
        program_info = None
        for goal, programs in app.training_programs.items():
            for program in programs:
                if program['id'] == current_program_id:
                    program_info = program
                    break
            if program_info:
                break
        
        if program_info:
            level_info = app.levels.get(program_info['level'], {})
            st.sidebar.markdown(f"""
            <div style='text-align: center; margin: 1rem 0;'>
                <span class='sport-icon'>📋</span>
                <h5>Текущая программа</h5>
                <p><strong>{program_info['name']}</strong></p>
                <span class='goal-badge {level_info.get("color", "level-beginner")}'>
                    {level_info.get('name', 'Начальный')}
                </span>
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
        
        # Кнопки для самообучающейся системы
        if st.session_state.current_user == "admin" or st.session_state.current_user.endswith("_admin"):
            st.markdown("### ⚙️ Администрирование")
            
            if st.button("🔄 Дообучить модель"):
                with st.spinner("Анализ отзывов и дообучение..."):
                    success, message = app.retrain_model_with_feedback()
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.warning(message)
            
            if st.button("📊 Статистика модели"):
                model_info = app.get_model_info()
                with st.expander("Информация о модели", expanded=True):
                    st.write(f"**Тип модели:** {model_info.get('model_type', 'Неизвестно')}")
                    st.write(f"**Количество признаков:** {model_info.get('feature_count', 0)}")
                    st.write(f"**Классы:** {', '.join(model_info.get('classes', []))}")
                    
                    # Статистика по отзывам
                    feedback_file = os.path.join(app.data_dir, 'user_feedback.csv')
                    if os.path.exists(feedback_file):
                        try:
                            feedback_df = pd.read_csv(feedback_file)
                            st.write(f"**Всего отзывов:** {len(feedback_df)}")
                            st.write(f"**Средний рейтинг:** {feedback_df['user_rating'].mean():.2f}")
                        except:
                            st.write("**Статистика отзывов:** Недоступна")
            
            if st.button("🧹 Очистить кэш модели"):
                try:
                    # Удаляем файлы модели
                    files_to_remove = [
                        'training_recommender.pkl',
                        'scaler.pkl', 
                        'label_encoder.pkl',
                        'training_data.npz',
                        'user_feedback.csv',
                        'retraining_log.json'
                    ]
                    removed = 0
                    for file in files_to_remove:
                        file_path = os.path.join(app.data_dir, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            removed += 1
                    
                    # Перезагружаем модель
                    app.init_ml_model()
                    st.success(f"✅ Удалено {removed} файлов. Модель перезагружена.")
                except Exception as e:
                    st.error(f"Ошибка очистки: {e}")
        
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
            # Персональная информация - ВНУТРИ прямоугольников
            personal_info = user_profile.get('personal_info', {})
            goals = user_profile.get('goals', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                st.markdown('<div class="progress-label">Текущий вес</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-metric">{personal_info.get("weight", 0)} кг</div>', unsafe_allow_html=True)
                st.markdown('<div class="progress-label">Желаемый вес</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-metric">{goals.get("target_weight", 0)} кг</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                bmi = user_profile.get('bmi', 0)
                bmi_category = user_profile.get('bmi_category', '')
                st.markdown('<div class="progress-label">ИМТ</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-metric">{bmi:.1f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-label">Категория: {bmi_category}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                calories_needed, tdee = app.calculate_calories_needed(user_profile)
                st.markdown('<div class="progress-label">Калории в день</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-metric">{calories_needed}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-label">Расход: {tdee} ккал</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Если у пользователя есть текущая программа
            if user_profile.get('current_program'):
                st.markdown("### 🏃 Текущая программа тренировок")
                
                current_program_id = user_profile['current_program']
                current_program = None
                for goal, programs in app.training_programs.items():
                    for program in programs:
                        if program['id'] == current_program_id:
                            current_program = program
                            break
                    if current_program:
                        break
                
                if current_program:
                    level_info = app.levels.get(current_program['level'], {})
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {current_program['name']}")
                        st.markdown(f"**Уровень:** <span class='goal-badge {level_info.get("color", "level-beginner")}'>{level_info.get('name', 'Начальный')}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Продолжительность:** {current_program['duration_weeks']} недель")
                        st.markdown(f"**Тренировок в неделю:** {current_program['sessions_per_week']}")
                    
                    with col2:
                        # АКТИВНАЯ кнопка для просмотра тренировок
                        if st.button("📋 Показать тренировки", use_container_width=True, key="show_current_program"):
                            st.session_state.show_program_details = current_program_id
                            st.rerun()
                    
                    st.markdown("**Расписание:**")
                    for session in current_program.get('schedule', []):
                        st.markdown(f"- {session}")
            
            # Рекомендуемые программы на основе ML с системой обратной связи
            st.markdown("### 🎯 Персональные рекомендации ИИ")
            
            recommended_programs = app.recommend_programs_based_on_profile(user_profile, display_feedback=True)
            
            if recommended_programs:
                for program in recommended_programs:
                    with st.container():
                        level_info = app.levels.get(program['level'], {})
                        
                        # Получаем информацию об активностях
                        activity_icons = ""
                        for activity_id in program.get('activities', []):
                            activity = app.activity_types.get(activity_id, {})
                            activity_icons += f"{activity.get('icon', '🏃')} "
                        
                        st.markdown(f"""
                        <div class="training-card">
                            <h3>{activity_icons} {program['name']}</h3>
                            <p><strong>Уровень:</strong> <span class='goal-badge {level_info.get("color", "level-beginner")}'>{level_info.get('name', 'Начальный')}</span> | <strong>Продолжительность:</strong> {program['duration_weeks']} недель</p>
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
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # АКТИВНАЯ кнопка для выбора программы
                            if st.button(f"🎯 Выбрать программу", key=f"select_{program['id']}", use_container_width=True):
                                if app.set_current_program(st.session_state.current_user, program['id']):
                                    st.success(f"✅ Программа '{program['name']}' выбрана!")
                                    st.rerun()
                                else:
                                    st.error("❌ Ошибка при выборе программы")
                        
                        with col2:
                            # АКТИВНАЯ кнопка для просмотра деталей
                            if st.button(f"📋 Посмотреть тренировки", key=f"details_{program['id']}", use_container_width=True):
                                st.session_state.show_program_details = program['id']
                                st.rerun()
                        
                        # --- СИСТЕМА ОБРАТНОЙ СВЯЗИ (ИСПРАВЛЕННАЯ) ---
                        st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                        st.caption("**Помогите нам стать лучше!** Оцените эту рекомендацию:")
                        
                        feedback_key = f"feedback_{program['id']}"
                        
                        # Используем session_state для хранения временного рейтинга
                        if feedback_key not in st.session_state.rating_temp:
                            st.session_state.rating_temp[feedback_key] = None
                        
                        feedback_cols = st.columns(5)
                        ratings = ["🤬", "😞", "😐", "🙂", "😍"]
                        ratings_values = [1, 2, 3, 4, 5]
                        
                        rating_submitted = False
                        
                        for idx, (col, emoji, rating_val) in enumerate(zip(feedback_cols, ratings, ratings_values)):
                            with col:
                                # Создаем уникальный ключ для каждой кнопки
                                button_key = f"rating_{program['id']}_{rating_val}"
                                
                                if st.button(emoji, key=button_key, use_container_width=True):
                                    st.session_state.rating_temp[feedback_key] = rating_val
                                    
                                    # Если оценка низкая, спрашиваем почему
                                    if rating_val <= 2:
                                        st.session_state.feedback_submitted[feedback_key] = False
                                        st.rerun()
                                    else:
                                        # Для высоких оценок сразу сохраняем
                                        success, message = app.collect_feedback(
                                            st.session_state.current_user,
                                            program['id'],
                                            rating_val,
                                            user_profile.get('goals', {}).get('primary_goal', 'weight_loss'),
                                            None,
                                            f"Оценка: {rating_val}/5"
                                        )
                                        if success:
                                            st.success("Спасибо за вашу оценку! 👍")
                                            st.session_state.feedback_submitted[feedback_key] = True
                                        else:
                                            st.error(message)
                                        st.rerun()
                        
                        # Если выбран низкий рейтинг (1-2), показываем форму для комментария
                        if (feedback_key in st.session_state.rating_temp and 
                            st.session_state.rating_temp[feedback_key] is not None and
                            st.session_state.rating_temp[feedback_key] <= 2 and
                            not st.session_state.feedback_submitted.get(feedback_key, False)):
                            
                            st.markdown("---")
                            st.write("**Пожалуйста, укажите почему:**")
                            
                            with st.form(key=f"low_rating_form_{program['id']}"):
                                actual_goal = st.selectbox(
                                    "Какая была бы более подходящая цель?",
                                    list(app.goals.keys()),
                                    format_func=lambda x: app.goals[x]['name'],
                                    key=f"actual_goal_{program['id']}"
                                )
                                comment = st.text_area("Дополнительные комментарии:", key=f"comment_{program['id']}")
                                
                                if st.form_submit_button("Отправить подробный отзыв"):
                                    success, message = app.collect_feedback(
                                        st.session_state.current_user,
                                        program['id'],
                                        st.session_state.rating_temp[feedback_key],
                                        user_profile.get('goals', {}).get('primary_goal', 'weight_loss'),
                                        actual_goal,
                                        comment
                                    )
                                    if success:
                                        st.success("Спасибо за подробный отзыв! Это очень поможет улучшить рекомендации.")
                                        st.session_state.feedback_submitted[feedback_key] = True
                                    else:
                                        st.error(message)
                                    st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("""
                💡 **Рекомендации появятся после заполнения анкеты.**
                
                Наш ИИ анализирует ваши данные и подбирает оптимальные тренировочные программы.
                Помогите нам стать лучше - оценивайте рекомендации!
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
            # Показываем текущую программу
            if user_profile.get('current_program'):
                current_program_id = user_profile['current_program']
                current_program = None
                for goal, programs in app.training_programs.items():
                    for program in programs:
                        if program['id'] == current_program_id:
                            current_program = program
                            break
                    if current_program:
                        break
                
                if current_program:
                    st.markdown("### 🏃 Текущая программа")
                    level_info = app.levels.get(current_program['level'], {})
                    
                    with st.expander(f"📋 {current_program['name']} ({level_info.get('name', 'Начальный')})", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Описание:** {current_program['description']}")
                            st.write(f"**Продолжительность:** {current_program['duration_weeks']} недель")
                            st.write(f"**Тренировок в неделю:** {current_program['sessions_per_week']}")
                            st.write(f"**Длительность тренировки:** {current_program['session_duration']} минут")
                            
                            st.write("**Расписание:**")
                            for session in current_program.get('schedule', []):
                                st.write(f"- {session}")
                        
                        with col2:
                            # Показываем иконки активностей
                            st.write("**Активности:**")
                            for activity_id in current_program.get('activities', []):
                                activity = app.activity_types.get(activity_id, {})
                                st.write(f"{activity.get('icon', '🏃')} {activity.get('name', activity_id)}")
                            
                            if st.button("📋 Показать тренировки", key="show_current_program_workouts"):
                                st.session_state.show_program_details = current_program_id
                                st.rerun()
            
            # Разделитель
            st.markdown("---")
            
            # Показываем все программы для цели пользователя
            goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
            goal_programs = app.training_programs.get(goal, [])
            
            if goal_programs:
                st.markdown(f"### 📊 Программы для вашей цели ({app.goals.get(goal, {}).get('name', 'Похудение')})")
                
                # Фильтр по уровню
                level_filter = st.selectbox(
                    "Фильтр по уровню:",
                    ["Все уровни", "Начальный", "Средний", "Продвинутый", "Профи"],
                    key="program_level_filter"
                )
                
                filtered_programs = []
                for program in goal_programs:
                    level_name = app.levels.get(program['level'], {}).get('name', 'Начальный')
                    if level_filter == "Все уровни" or level_name == level_filter:
                        filtered_programs.append(program)
                
                if filtered_programs:
                    st.success(f"📊 Найдено {len(filtered_programs)} программ")
                    
                    for program in filtered_programs:
                        level_info = app.levels.get(program['level'], {})
                        
                        with st.expander(f"{program['name']} ({level_info.get('name', 'Начальный')})"):
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
                                
                                col_btn1, col_btn2 = st.columns(2)
                                with col_btn1:
                                    if st.button(f"✅ Выбрать", key=f"select_program_{program['id']}"):
                                        if app.set_current_program(st.session_state.current_user, program['id']):
                                            st.success(f"Программа '{program['name']}' выбрана!")
                                            st.rerun()
                                with col_btn2:
                                    if st.button(f"📋", key=f"view_program_{program['id']}"):
                                        st.session_state.show_program_details = program['id']
                                        st.rerun()
                else:
                    st.info(f"Нет программ для выбранного уровня '{level_filter}'")
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
            
            # Если есть текущая программа, добавляем информацию о ней
            current_program = user_profile.get('current_program')
            program_id = None
            day = None
            if current_program:
                program_id = current_program
                day_options = [f"День {i}" for i in range(1, 8)]
                day = st.selectbox("День программы:", options=day_options, key="workout_day")
            
            submit_button = st.form_submit_button("💾 Сохранить тренировку", use_container_width=True)
            
            if submit_button:
                success, message = app.add_workout(
                    st.session_state.current_user, 
                    workout_type_clean, 
                    duration, 
                    intensity, 
                    notes,
                    program_id,
                    day
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                    # Сбрасываем просмотр программы
                    if 'show_program_details' in st.session_state:
                        st.session_state.show_program_details = None
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
            📋 **Программа начата** - Начало тренировочной программы
            💬 **Эксперт по обратной связи** - Оставьте 5 отзывов
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
                
                # Кнопка отправки формы
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

# Обработка просмотра деталей программы (ИСПРАВЛЕННАЯ)
if st.session_state.get('show_program_details'):
    program_id = st.session_state.show_program_details
    
    # Находим программу
    program_info = None
    for goal, programs in app.training_programs.items():
        for program in programs:
            if program['id'] == program_id:
                program_info = program
                break
        if program_info:
            break
    
    if program_info:
        # Создаем модальное окно
        st.markdown("---")
        st.markdown(f"### 📋 {program_info['name']}")
        
        level_info = app.levels.get(program_info['level'], {})
        st.markdown(f"**Уровень:** <span class='goal-badge {level_info.get("color", "level-beginner")}'>{level_info.get('name', 'Начальный')}</span>", unsafe_allow_html=True)
        
        # Получаем все дни тренировок
        workout_days = app.get_all_workout_days(program_id)
        
        if workout_days:
            # Выбор дня тренировки
            if st.session_state.get('selected_day') and st.session_state.selected_day in workout_days:
                selected_day = st.session_state.selected_day
            else:
                selected_day = workout_days[0]
            
            # Создаем табы для дней
            tabs = st.tabs([f"День {i+1}" for i in range(len(workout_days))])
            
            for i, (tab, day_key) in enumerate(zip(tabs, workout_days)):
                with tab:
                    # Получаем упражнения для дня
                    exercises = app.get_exercises_for_program(program_id, day_key)
                    
                    if exercises:
                        st.markdown(f"#### {exercises.get('title', f'Тренировка {i+1}')}")
                        
                        # Видео тренировки
                        if 'video_url' in exercises:
                            st.markdown(f"""
                            <div style='margin: 1rem 0; padding: 1rem; background: #f0f8ff; border-radius: 10px;'>
                                <h5>🎥 Видео тренировки</h5>
                                <p>{exercises.get('video_description', 'Полная тренировка')}</p>
                                <a href='{exercises['video_url']}' target='_blank' class='video-link'>
                                    📺 Смотреть тренировку на YouTube
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Разминка
                        if 'warmup' in exercises:
                            st.markdown(f"**🔥 Разминка:** {exercises['warmup']}")
                        
                        # Упражнения
                        st.markdown("##### 📋 Упражнения:")
                        for j, exercise in enumerate(exercises.get('exercises', [])):
                            with st.container():
                                st.markdown(f"""
                                <div class="exercise-item">
                                    <h5>{j+1}. {exercise.get('name', 'Упражнение')}</h5>
                                    <p><strong>Тип:</strong> {exercise.get('type', 'Общее')}</p>
                                """, unsafe_allow_html=True)
                                
                                if 'duration' in exercise:
                                    st.markdown(f"<p><strong>Длительность:</strong> {exercise['duration']}</p>", unsafe_allow_html=True)
                                if 'sets' in exercise and 'reps' in exercise:
                                    st.markdown(f"<p><strong>Подходы/Повторения:</strong> {exercise['sets']} × {exercise['reps']}</p>", unsafe_allow_html=True)
                                if 'rest' in exercise:
                                    st.markdown(f"<p><strong>Отдых:</strong> {exercise['rest']}</p>", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Заминка
                        if 'cooldown' in exercises:
                            st.markdown(f"**🧘 Заминка:** {exercises['cooldown']}")
                        
                        # Кнопка для добавления этой тренировки (ИСПРАВЛЕННАЯ)
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            # АКТИВНАЯ кнопка для добавления тренировки
                            add_key = f"add_workout_{day_key}_{i}"
                            if st.button(f"➕ Добавить тренировку День {i+1}", 
                                       use_container_width=True, 
                                       key=add_key):
                                # Сохраняем информацию о выбранной программе и дне
                                st.session_state.selected_program_for_workout = program_id
                                st.session_state.selected_day_for_workout = f"День {i+1}"
                                st.session_state.selected_workout_title = exercises.get('title', f'Тренировка {i+1}')
                                # Переходим на страницу добавления тренировки
                                st.session_state.current_page = "➕ Добавить тренировку"
                                st.rerun()
                        with col2:
                            if st.button("❌ Закрыть", use_container_width=True, key=f"close_{day_key}"):
                                st.session_state.show_program_details = None
                                st.session_state.selected_day = None
                                st.rerun()
                    else:
                        st.info(f"Для дня {i+1} пока нет конкретных упражнений.")
            
        else:
            st.info("Для этой программы пока нет конкретных тренировок.")
            if st.button("❌ Закрыть", use_container_width=True):
                st.session_state.show_program_details = None
                st.session_state.selected_day = None
                st.rerun()

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💪 <strong>Фитнес Помощник v10.0</strong> | Самообучающаяся система рекомендаций</p>
    <p>Ваш персональный ИИ-тренер, который становится умнее с каждым отзывом!</p>
</div>
""", unsafe_allow_html=True)
