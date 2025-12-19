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

# Новый CSS стиль как на изображении
st.markdown("""
<style>
    /* Основные стили как на картинке */
    :root {
        --primary: #2ecc71;
        --secondary: #3498db;
        --accent: #9b59b6;
        --dark: #1a1a2e;
        --dark-light: #2d2d44;
        --text: #ecf0f1;
        --text-secondary: #bdc3c7;
        --card-bg: rgba(45, 45, 68, 0.7);
        --border-radius: 16px;
    }
    
    .stApp {
        background-color: var(--dark);
        color: var(--text);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(46, 204, 113, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(155, 89, 182, 0.1) 0%, transparent 20%);
    }
    
    /* Заголовки */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid var(--primary);
        padding-left: 12px;
    }
    
    /* Карточки как на изображении */
    .modern-card {
        background-color: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(52, 152, 219, 0.1) 100%);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .activity-card {
        background: linear-gradient(135deg, var(--primary), #27ae60);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        color: white;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.2);
    }
    
    .nutrition-card {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        border-radius: var(--border-radius);
        padding: 1.2rem;
        color: white;
    }
    
    /* Крупные цифры */
    .big-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(to right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        line-height: 1;
    }
    
    .medium-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
    }
    
    /* Прогресс-бары */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
    }
    
    .progress-protein { background-color: var(--primary); width: 78%; }
    .progress-carbs { background-color: var(--secondary); width: 65%; }
    .progress-fat { background-color: var(--accent); width: 32%; }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    
    /* Пользовательская карточка */
    .user-widget {
        background: linear-gradient(135deg, var(--dark-light), var(--dark));
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Время и метки */
    .time-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Иконки */
    .icon-large {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Анимации */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Сайдбар */
    .css-1d391kg {
        background-color: rgba(26, 26, 46, 0.9);
        backdrop-filter: blur(10px);
    }
    
    /* Ползунки и инпуты */
    .stSlider > div > div > div {
        background: var(--primary);
    }
    
    /* Таблицы */
    .dataframe {
        background-color: var(--card-bg) !important;
        color: var(--text) !important;
    }
    
    .dataframe th {
        background-color: var(--dark-light) !important;
        color: var(--text) !important;
    }
    
    .dataframe td {
        color: var(--text) !important;
    }
    
    /* Экспандеры */
    .streamlit-expanderHeader {
        background-color: var(--dark-light) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }
    
    /* Текстовые поля */
    .stTextInput > div > div > input {
        background-color: var(--dark-light);
        color: var(--text);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Селект боксы */
    .stSelectbox > div > div > div {
        background-color: var(--dark-light);
        color: var(--text);
    }
    
    /* Убираем стандартные отступы */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
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
                        'День 1: Утренная йога 20 мин',
                        'День 2: Вечерняя растяжка 30 мин',
                        'День 3: Йога для спины 25 мин',
                        'День 4: Отдых',
                        'День 5: Полная сессия йоги 30 мин'
                    ],
                    'workouts': {
                        'day1': {
                            'title': 'Утренная йога',
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
                                {'type': 'yoga', 'name': 'Поза ребенка', 'duration': '5 минут'},
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
            st.warning("Используются рекомендации по умолчанию")
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
            'sedentary': 1.2,      # Сидячий образ жизни
            'light': 1.375,        # Легкая активность 1-3 раза в неделю
            'moderate': 1.55,      # Умеренная активность 3-5 раз в неделю
            'active': 1.725,       # Высокая активность 6-7 раз в неделю
            'very_active': 1.9,    # Очень высокая активность,
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
            if abs(current - target) <= 2:  # Достигли цели в пределах 2 кг
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
    if 'show_program_details' not in st.session_state:
        st.session_state.show_program_details = None
    if 'selected_day' not in st.session_state:
        st.session_state.selected_day = None

initialize_session_state()

# Страница входа/регистрации
if not st.session_state.authenticated:
    # Стилизованный заголовок
    col_header = st.columns([3, 1])
    with col_header[0]:
        st.markdown('<h1 class="main-title">💪 Фитнес Помощник</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.2rem; color: #bdc3c7;">Персональный тренер для любого вида фитнеса</p>', unsafe_allow_html=True)
    
    # Карточка входа
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    if st.session_state.show_login:
        # Форма входа в новый дизайне
        with st.form("login_form"):
            cols = st.columns([2, 1, 2])
            with cols[0]:
                st.markdown('<h3>🔐 Вход в систему</h3>', unsafe_allow_html=True)
            
            with cols[2]:
                if st.form_submit_button("📝 Регистрация", use_container_width=True):
                    st.session_state.show_login = False
                    st.session_state.show_registration = True
                    st.rerun()
            
            login_username = st.text_input("Логин:", placeholder="Введите ваш логин")
            login_password = st.text_input("Пароль:", type="password", placeholder="Введите ваш пароль")
            
            col_btn = st.columns(2)
            with col_btn[0]:
                login_submitted = st.form_submit_button("Войти", use_container_width=True)
            
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
        # Форма регистрации в новом дизайне
        with st.form("register_form"):
            st.markdown('<h3>📝 Регистрация</h3>', unsafe_allow_html=True)
            
            reg_username = st.text_input("Логин:", placeholder="Придумайте логин")
            reg_password = st.text_input("Пароль:", type="password", placeholder="Придумайте пароль")
            reg_confirm = st.text_input("Подтвердите пароль:", type="password", placeholder="Повторите пароль")
            
            col_btn = st.columns(2)
            with col_btn[0]:
                reg_submitted = st.form_submit_button("Зарегистрироваться", use_container_width=True)
            with col_btn[1]:
                if st.form_submit_button("← Назад к входу", use_container_width=True):
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# Анкета после регистрации/первого входа
elif st.session_state.show_questionnaire:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="main-title">📝 Давайте познакомимся!</h2>', unsafe_allow_html=True)
    
    with st.form("questionnaire_form"):
        st.markdown('<h3 class="section-header">📊 Личные данные</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Возраст:", min_value=10, max_value=100, value=25, key="q_age")
            height = st.number_input("Рост (см):", min_value=100, max_value=250, value=170, key="q_height")
        with col2:
            weight = st.number_input("Текущий вес (кг):", min_value=30, max_value=200, value=70, key="q_weight")
            gender = st.selectbox("Пол:", ["Женский", "Мужской"], key="q_gender")
        
        st.markdown('<h3 class="section-header">🎯 Ваши цели</h3>', unsafe_allow_html=True)
        
        primary_goal = st.selectbox("Основная цель:", 
                                  ["Похудение", "Набор мышечной массы", "Улучшение выносливости", 
                                   "Развитие гибкости", "Общее оздоровление"], key="q_goal")
        
        target_weight = st.number_input("Желаемый вес (кг):", min_value=30, max_value=200, value=65, key="q_target_weight")
        
        st.markdown('<h3 class="section-header">🏋️‍♀️ Предпочитаемые виды активности</h3>', unsafe_allow_html=True)
        
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
        
        st.markdown('<h3 class="section-header">📊 Уровень активности</h3>', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # ОСНОВНОЕ ПРИЛОЖЕНИЕ (после входа)
    
    # Загрузка профиля пользователя
    user_profile = app.load_user_profile(st.session_state.current_user)
    
    # Боковая панель в новом стиле
    with st.sidebar:
        # Карточка пользователя
        st.markdown('<div class="user-widget">', unsafe_allow_html=True)
        st.markdown(f"### 👤 {st.session_state.current_user}")
        
        # Показываем цель пользователя
        if user_profile.get('goals', {}).get('primary_goal'):
            goal_info = app.goals.get(user_profile['goals']['primary_goal'], {})
            if goal_info:
                st.markdown(f"""
                <div style='text-align: center; margin: 1rem 0;'>
                    <span style='font-size: 2rem;'>{goal_info['icon']}</span>
                    <p><strong>{goal_info['name']}</strong></p>
                    <p style='font-size: 0.8rem; color: #bdc3c7;'>{goal_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Навигация
        st.markdown('<h3 class="section-header">Навигация</h3>', unsafe_allow_html=True)
        
        page = st.radio(
            "Выберите раздел:",
            ["📊 Главная", "🎯 Мои программы", "➕ Добавить тренировку", "📈 Мой прогресс", "🏆 Достижения", "👤 Мой профиль"],
            index=["📊 Главная", "🎯 Мои программы", "➕ Добавить тренировку", "📈 Мой прогресс", "🏆 Достижения", "👤 Мой профиль"].index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.markdown("---")
        
        # Статистика
        stats = app.get_statistics(st.session_state.current_user)
        if stats:
            st.markdown('<h3 class="section-header">Статистика</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='big-number'>{stats['total_workouts']}</div><div class='metric-label'>Тренировок</div></div>", unsafe_allow_html=True)
            with col2:
                minutes = int(stats['total_minutes'])
                st.markdown(f"<div class='metric-card'><div class='big-number'>{minutes}</div><div class='metric-label'>Минут</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("✏️ Редактировать анкету", use_container_width=True):
            st.session_state.show_questionnaire = True
            st.rerun()
        
        if st.button("🚪 Выйти", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = ""
            st.rerun()

    # Главная страница
    if st.session_state.current_page == "📊 Главная":
        # Верхняя часть - приветствие и цель
        col_top = st.columns([3, 1])
        with col_top[0]:
            st.markdown(f'<h2 class="main-title">Привет, {st.session_state.current_user}!</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #bdc3c7;">{datetime.now().strftime("%A, %d %B")}</p>', unsafe_allow_html=True)
        
        with col_top[1]:
            if user_profile.get('goals', {}).get('primary_goal'):
                goal_info = app.goals.get(user_profile['goals']['primary_goal'], {})
                st.markdown(f"""
                <div class="activity-card">
                    <div style='font-size: 1.2rem; font-weight: bold;'>{goal_info.get('icon', '🎯')}</div>
                    <div style='font-size: 1.1rem;'>Достигнуть цели</div>
                </div>
                """, unsafe_allow_html=True)
        
        if not user_profile.get('questionnaire_completed', False):
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.warning("""
            ⚠️ **Анкета не заполнена!**
            
            Для получения персонализированных рекомендаций заполните анкету.
            """)
            if st.button("📝 Заполнить анкету", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Блок с активностью (как на картинке)
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            col_activity = st.columns([3, 1])
            with col_activity[0]:
                st.markdown('<h3 class="section-header">🏃 Текущая активность</h3>', unsafe_allow_html=True)
                # Берем последнюю тренировку
                workouts = app.get_all_workouts(st.session_state.current_user)
                if not workouts.empty:
                    last_workout = workouts.iloc[0]
                    st.markdown(f"**{last_workout['workout_type']}**")
                    st.markdown(f"<p style='color: #bdc3c7;'>{last_workout.get('notes', '')}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("**Нет недавних тренировок**")
                    st.markdown('<p style="color: #bdc3c7;">Добавьте первую тренировку!</p>', unsafe_allow_html=True)
            
            with col_activity[1]:
                if not workouts.empty:
                    last_workout = workouts.iloc[0]
                    duration = last_workout['duration']
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem;'>
                        <div style='font-size: 2.5rem; font-weight: bold; color: #9b59b6;'>{duration}</div>
                        <div style='font-size: 0.9rem; color: #bdc3c7;'>минут</div>
                        <div style='font-size: 0.8rem; color: #95a5a6; margin-top: 0.5rem;'>Сегодня {datetime.now().strftime("%H:%M")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='text-align: center; padding: 1rem;'>
                        <div style='font-size: 2.5rem; font-weight: bold; color: #9b59b6;'>0</div>
                        <div style='font-size: 0.9rem; color: #bdc3c7;'>минут</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Метрики (шаги, активность, вода, сон) - как на картинке
            st.markdown('<h3 class="section-header">📊 Сегодняшние метрики</h3>', unsafe_allow_html=True)
            
            col_metrics = st.columns(4)
            
            with col_metrics[0]:
                # Шаги (симулируем на основе тренировок)
                step_count = workouts['duration'].sum() * 100 if not workouts.empty else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="big-number">{int(step_count):,}</div>
                    <div class="metric-label">Шаги</div>
                    <div class="time-label">Сегодня 11:45</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metrics[1]:
                # Активность
                activity_minutes = workouts['duration'].sum() if not workouts.empty else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="medium-number">{int(activity_minutes)}</div>
                    <div class="metric-label">Минут активности</div>
                    <div class="time-label">Сегодня 21:00</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metrics[2]:
                # Вода
                st.markdown(f"""
                <div class="metric-card">
                    <div class="medium-number">2.1</div>
                    <div class="metric-label">Литров воды</div>
                    <div class="time-label">Сегодня 9:00</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metrics[3]:
                # Сон
                st.markdown(f"""
                <div class="metric-card">
                    <div class="medium-number">7:25</div>
                    <div class="metric-label">Часов сна</div>
                    <div class="time-label">Сегодня 7:00</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Питание - адаптированное под новый дизайн
            st.markdown('<h3 class="section-header">🍽️ Питание сегодня</h3>', unsafe_allow_html=True)
            
            col_nutrition = st.columns(3)
            
            meals = [
                {"name": "Завтрак", "calories": "420 ккал", "items": ["Овсянка с ягодами", "Омлет", "Кофе"]},
                {"name": "Обед", "calories": "680 ккал", "items": ["Куриная грудка", "Гречка", "Салат"]},
                {"name": "Ужин", "calories": "520 ккал", "items": ["Лосось", "Брокколи", "Авокадо"]}
            ]
            
            for i, meal in enumerate(meals):
                with col_nutrition[i]:
                    st.markdown(f"""
                    <div class="nutrition-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h4>{meal['name']}</h4>
                            <span style="color: #2ecc71; font-weight: bold;">{meal['calories']}</span>
                        </div>
                        <ul style="padding-left: 1.2rem; margin: 0;">
                    """, unsafe_allow_html=True)
                    
                    for item in meal['items']:
                        st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Макронутриенты
            st.markdown('<h3 class="section-header">⚖️ Макронутриенты</h3>', unsafe_allow_html=True)
            
            col_macros = st.columns(3)
            
            macros = [
                {"name": "Белки", "value": "125г", "target": "150г", "progress_class": "progress-protein"},
                {"name": "Углеводы", "value": "210г", "target": "230г", "progress_class": "progress-carbs"},
                {"name": "Жиры", "value": "55г", "target": "70г", "progress_class": "progress-fat"}
            ]
            
            for i, macro in enumerate(macros):
                with col_macros[i]:
                    st.markdown(f"""
                    <div class="modern-card">
                        <h4>{macro['name']}</h4>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>{macro['value']}</span>
                            <span style="color: #bdc3c7;">из {macro['target']}</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-fill {macro['progress_class']}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Рекомендуемые программы
            st.markdown('<h3 class="section-header">🎯 Персональные рекомендации</h3>', unsafe_allow_html=True)
            
            recommended_programs = app.recommend_programs_based_on_profile(user_profile)
            
            if recommended_programs:
                for program in recommended_programs:
                    with st.container():
                        level_info = app.levels.get(program['level'], {})
                        
                        # Получаем иконки активностей
                        activity_icons = ""
                        for activity_id in program.get('activities', []):
                            activity = app.activity_types.get(activity_id, {})
                            activity_icons += f"{activity.get('icon', '🏃')} "
                        
                        st.markdown(f"""
                        <div class="modern-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h4>{activity_icons} {program['name']}</h4>
                                    <p style="color: #bdc3c7; margin: 0.5rem 0;">{program['description']}</p>
                                    <p><strong>Продолжительность:</strong> {program['duration_weeks']} недель</p>
                                    <p><strong>Тренировок в неделю:</strong> {program['sessions_per_week']}</p>
                                </div>
                                <div style="text-align: right;">
                                    <span style="background: #3498db; color: white; padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.9rem;">
                                        {level_info.get('name', 'Начальный')}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_btns = st.columns(2)
                        with col_btns[0]:
                            if st.button(f"🎯 Выбрать программу", key=f"select_{program['id']}", use_container_width=True):
                                if app.set_current_program(st.session_state.current_user, program['id']):
                                    st.success(f"✅ Программа '{program['name']}' выбрана!")
                                    st.rerun()
                        with col_btns[1]:
                            if st.button(f"📋 Посмотреть тренировки", key=f"details_{program['id']}", use_container_width=True):
                                st.session_state.show_program_details = program['id']
                                st.rerun()
            
            # Популярные упражнения
            st.markdown('<h3 class="section-header">🔥 Популярные упражнения</h3>', unsafe_allow_html=True)
            
            col_exercises = st.columns(4)
            
            exercises = [
                {"name": "Приседания", "count": "15 тренировок", "progress": 75},
                {"name": "Жим лежа", "count": "11 тренировок", "progress": 60},
                {"name": "Тяга", "count": "8 тренировок", "progress": 40},
                {"name": "Планка", "count": "22 тренировки", "progress": 90}
            ]
            
            for i, exercise in enumerate(exercises):
                with col_exercises[i]:
                    st.markdown(f"""
                    <div class="modern-card">
                        <h4>{exercise['name']}</h4>
                        <p style="color: #bdc3c7; font-size: 0.9rem;">{exercise['count']}</p>
                        <div class="progress-container">
                            <div class="progress-fill" style="width: {exercise['progress']}%; background-color: #2ecc71;"></div>
                        </div>
                        <div style="text-align: right; margin-top: 0.5rem;">
                            <span style="color: #bdc3c7; font-size: 0.9rem;">{exercise['progress']}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Мои программы
    elif st.session_state.current_page == "🎯 Мои программы":
        st.markdown('<h2 class="main-title">🎯 Мои тренировочные программы</h2>', unsafe_allow_html=True)
        
        if not user_profile.get('questionnaire_completed', False):
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.warning("Заполните анкету для получения персональных программ")
            st.markdown('</div>', unsafe_allow_html=True)
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
                    st.markdown('<div class="activity-card">', unsafe_allow_html=True)
                    level_info = app.levels.get(current_program['level'], {})
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### 🏃 {current_program['name']}")
                        st.markdown(f"**Уровень:** {level_info.get('name', 'Начальный')}")
                        st.markdown(f"**Описание:** {current_program['description']}")
                    with col2:
                        if st.button("📋 Показать тренировки", use_container_width=True):
                            st.session_state.show_program_details = current_program_id
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Разделитель
            st.markdown("---")
            
            # Показываем все программы для цели пользователя
            goal = user_profile.get('goals', {}).get('primary_goal', 'weight_loss')
            goal_programs = app.training_programs.get(goal, [])
            
            if goal_programs:
                st.markdown(f'<h3 class="section-header">📊 Программы для вашей цели</h3>', unsafe_allow_html=True)
                
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
                    for program in filtered_programs:
                        level_info = app.levels.get(program['level'], {})
                        
                        with st.expander(f"{program['name']} ({level_info.get('name', 'Начальный')})", expanded=False):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Описание:** {program['description']}")
                                st.markdown(f"**Продолжительность:** {program['duration_weeks']} недель")
                                st.markdown(f"**Тренировок в неделю:** {program['sessions_per_week']}")
                                st.markdown(f"**Длительность тренировки:** {program['session_duration']} минут")
                                
                                st.markdown("**Расписание:**")
                                for session in program.get('schedule', []):
                                    st.markdown(f"- {session}")
                            
                            with col2:
                                # Показываем иконки активностей
                                st.markdown("**Активности:**")
                                for activity_id in program.get('activities', []):
                                    activity = app.activity_types.get(activity_id, {})
                                    st.markdown(f"{activity.get('icon', '🏃')} {activity.get('name', activity_id)}")
                                
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
        st.markdown('<h2 class="main-title">➕ Добавить тренировку</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        with st.form("add_workout_form"):
            # Проверяем, есть ли предзаполненные данные из программы
            program_id = None
            day = None
            
            if 'selected_program_for_workout' in st.session_state:
                program_id = st.session_state.selected_program_for_workout
                day = st.session_state.get('selected_day_for_workout', 'День 1')
                
                # Показываем информацию о выбранной программе
                program_info = None
                for goal, programs in app.training_programs.items():
                    for program in programs:
                        if program['id'] == program_id:
                            program_info = program
                            break
                    if program_info:
                        break
                
                if program_info:
                    st.info(f"📋 Добавляем тренировку из программы: **{program_info['name']}** ({day})")
            
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
                
                # Если есть программа, подсказываем тип тренировки
                if program_info and program_info.get('activities'):
                    # Получаем первую активность из программы как рекомендуемую
                    first_activity_id = program_info['activities'][0] if program_info['activities'] else None
                    if first_activity_id:
                        activity = app.activity_types.get(first_activity_id, {})
                        default_option = f"{activity.get('icon', '🏃')} {activity.get('name', first_activity_id)}"
                        if default_option in workout_options:
                            workout_index = workout_options.index(default_option)
                            workout_type = st.selectbox(
                                "Вид тренировки:",
                                options=workout_options,
                                index=workout_index,
                                key="workout_type_select"
                            )
                        else:
                            workout_type = st.selectbox(
                                "Вид тренировки:",
                                options=workout_options,
                                key="workout_type_select"
                            )
                    else:
                        workout_type = st.selectbox(
                            "Вид тренировки:",
                            options=workout_options,
                            key="workout_type_select"
                        )
                else:
                    workout_type = st.selectbox(
                        "Вид тренировки:",
                        options=workout_options,
                        key="workout_type_select"
                    )
                workout_type_clean = workout_mapping[workout_type]
            else:
                # Если есть программа, подсказываем тип тренировки
                if program_info:
                    # Используем первую активность из программы как подсказку
                    if program_info.get('activities'):
                        first_activity_id = program_info['activities'][0]
                        activity = app.activity_types.get(first_activity_id, {})
                        default_text = activity.get('name', first_activity_id)
                    else:
                        default_text = ""
                    workout_type = st.text_input("Вид тренировки:", 
                                                value=default_text,
                                                placeholder="Например: Йога, Бег, Пилатес...", 
                                                key="workout_type_text")
                else:
                    workout_type = st.text_input("Вид тренировки:", 
                                                placeholder="Например: Йога, Бег, Пилатес...", 
                                                key="workout_type_text")
                workout_type_clean = workout_type
            
            col1, col2 = st.columns(2)
            with col1:
                # Если есть программа, подсказываем длительность
                if program_info and 'session_duration' in program_info:
                    default_duration = program_info['session_duration']
                else:
                    default_duration = 45
                    
                duration = st.number_input("Длительность (минут):", 
                                         min_value=5, 
                                         max_value=180, 
                                         value=default_duration, 
                                         key="workout_duration")
            with col2:
                intensity = st.select_slider(
                    "Интенсивность:",
                    options=["Очень легкая", "Легкая", "Средняя", "Высокая", "Очень высокая"],
                    value="Средняя",
                    key="workout_intensity"
                )
            
            # Если есть программа, подсказываем заметки
            if program_info:
                default_notes = f"Тренировка из программы: {program_info['name']}, {day}"
            else:
                default_notes = ""
                
            notes = st.text_area("Заметки:", 
                                value=default_notes,
                                placeholder="Как прошла тренировка? Что понравилось?", 
                                key="workout_notes")
            
            # Если есть текущая программа ИЛИ мы перешли из просмотра программы, показываем информацию о дне
            if program_id:
                # Показываем информацию о дне, но не даем выбирать
                st.info(f"📅 День программы: {day}")
                # Сохраняем эти значения для отправки
                final_program_id = program_id
                final_day = day
            elif user_profile.get('current_program'):
                # Если есть текущая программа пользователя, но мы не из просмотра
                current_program = user_profile.get('current_program')
                final_program_id = current_program
                day_options = [f"День {i}" for i in range(1, 8)]
                final_day = st.selectbox("День программы:", 
                                       options=day_options, 
                                       key="workout_day")
            else:
                final_program_id = None
                final_day = None
            
            col_submit, col_clear = st.columns(2)
            with col_submit:
                submit_button = st.form_submit_button("💾 Сохранить тренировку", use_container_width=True)
            
            if submit_button:
                success, message = app.add_workout(
                    st.session_state.current_user, 
                    workout_type_clean, 
                    duration, 
                    intensity, 
                    notes,
                    final_program_id,
                    final_day
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                    
                    # Очищаем временные данные сессии после успешного сохранения
                    if 'selected_program_for_workout' in st.session_state:
                        del st.session_state.selected_program_for_workout
                    if 'selected_day_for_workout' in st.session_state:
                        del st.session_state.selected_day_for_workout
                    
                    # Даем пользователю выбор
                    col_again, col_home = st.columns(2)
                    with col_again:
                        if st.button("➕ Добавить еще тренировку", use_container_width=True):
                            # Очищаем форму, оставляя на той же странице
                            st.rerun()
                    with col_home:
                        if st.button("🏠 На главную", use_container_width=True):
                            st.session_state.current_page = "📊 Главная"
                            st.rerun()
                else:
                    st.error(message)
            
            with col_clear:
                # Кнопка для очистки предзаполненных данных
                if program_id:
                    if st.form_submit_button("🗑️ Очистить программу", use_container_width=True):
                        # Очищаем временные данные
                        if 'selected_program_for_workout' in st.session_state:
                            del st.session_state.selected_program_for_workout
                        if 'selected_day_for_workout' in st.session_state:
                            del st.session_state.selected_day_for_workout
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Мой прогресс
    elif st.session_state.current_page == "📈 Мой прогресс":
        st.markdown('<h2 class="main-title">📈 Мой прогресс</h2>', unsafe_allow_html=True)
        
        # Статистика тренировок в новом стиле
        stats = app.get_statistics(st.session_state.current_user)
        workouts = app.get_all_workouts(st.session_state.current_user)
        
        if stats:
            col_stats = st.columns(4)
            with col_stats[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="big-number">{stats['total_workouts']}</div>
                    <div class="metric-label">Всего тренировок</div>
                </div>
                """, unsafe_allow_html=True)
            with col_stats[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="big-number">{int(stats['total_minutes'])}</div>
                    <div class="metric-label">Минут тренировок</div>
                </div>
                """, unsafe_allow_html=True)
            with col_stats[2]:
                avg_duration = stats['avg_duration'] if not pd.isna(stats['avg_duration']) else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="big-number">{int(avg_duration)}</div>
                    <div class="metric-label">Средняя длительность</div>
                </div>
                """, unsafe_allow_html=True)
            with col_stats[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="big-number">{stats.get('workout_streak', 0)}</div>
                    <div class="metric-label">Текущая серия</div>
                </div>
                """, unsafe_allow_html=True)
        
        # График тренировок
        if not workouts.empty:
            st.markdown('<h3 class="section-header">📊 График активности</h3>', unsafe_allow_html=True)
            
            # Группируем по дням
            workouts['date_only'] = workouts['date'].dt.date
            daily_workouts = workouts.groupby('date_only').agg({
                'duration': 'sum',
                'workout_type': 'count'
            }).reset_index()
            daily_workouts.columns = ['date', 'total_minutes', 'workout_count']
            
            # Стилизованный график
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # График 1: Длительность тренировок по дням
            ax1.bar(daily_workouts['date'], daily_workouts['total_minutes'], color='#2ecc71')
            ax1.set_title('Длительность тренировок по дням', fontsize=14, fontweight='bold', color='white')
            ax1.set_ylabel('Минуты', color='white')
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')
            ax1.set_facecolor('#1a1a2e')
            ax1.grid(True, alpha=0.3, color='white')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, color='white')
            
            # График 2: Количество тренировок по дням
            ax2.bar(daily_workouts['date'], daily_workouts['workout_count'], color='#3498db')
            ax2.set_title('Количество тренировок по дням', fontsize=14, fontweight='bold', color='white')
            ax2.set_ylabel('Тренировки', color='white')
            ax2.set_xlabel('Дата', color='white')
            ax2.tick_params(axis='x', colors='white')
            ax2.tick_params(axis='y', colors='white')
            ax2.set_facecolor('#1a1a2e')
            ax2.grid(True, alpha=0.3, color='white')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, color='white')
            
            fig.patch.set_facecolor('#1a1a2e')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Таблица последних тренировок
            st.markdown('<h3 class="section-header">📋 История тренировок</h3>', unsafe_allow_html=True)
            recent_workouts = workouts.head(10).copy()
            recent_workouts['date'] = recent_workouts['date'].dt.strftime('%d.%m.%Y %H:%M')
            # Стилизуем таблицу
            st.dataframe(recent_workouts[['date', 'workout_type', 'duration', 'intensity', 'notes']], 
                        use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.info("📝 У вас пока нет тренировок. Добавьте первую!")
            st.markdown('</div>', unsafe_allow_html=True)

    # Достижения
    elif st.session_state.current_page == "🏆 Достижения":
        st.markdown('<h2 class="main-title">🏆 Мои достижения</h2>', unsafe_allow_html=True)
        
        achievements = app.get_achievements(st.session_state.current_user)
        stats = app.get_statistics(st.session_state.current_user)
        
        if achievements:
            unlocked = [a for a in achievements if a.get('unlocked', False)]
            total = len(achievements)
            
            # Прогресс-бар достижений
            st.markdown('<h3 class="section-header">🎯 Прогресс достижений</h3>', unsafe_allow_html=True)
            if total > 0:
                progress = len(unlocked) / total * 100
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); border-radius: 10px; padding: 1rem; margin-bottom: 1rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>Получено достижений: {len(unlocked)}/{total}</span>
                        <span>{progress:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {progress}%; background-color: #f39c12;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Отображение достижений в новом стиле
            st.markdown('<h3 class="section-header">🏆 Полученные достижения</h3>', unsafe_allow_html=True)
            if unlocked:
                cols = st.columns(3)
                for i, achievement in enumerate(unlocked):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="nutrition-card">
                            <div style='text-align: center; font-size: 2rem; margin-bottom: 0.5rem;'>{achievement['icon']}</div>
                            <h4 style='text-align: center;'>{achievement['title']}</h4>
                            <p style='text-align: center; font-size: 0.9rem; color: #bdc3c7;'>{achievement['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Ближайшие цели
            if stats:
                st.markdown('<h3 class="section-header">🎯 Ближайшие цели</h3>', unsafe_allow_html=True)
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
                    st.markdown('<div class="activity-card">', unsafe_allow_html=True)
                    st.success("🎊 Все основные цели достигнуты! Вы настоящий чемпион! 🏆")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
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
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # Мой профиль
    elif st.session_state.current_page == "👤 Мой профиль":
        st.markdown('<h2 class="main-title">👤 Мой профиль</h2>', unsafe_allow_html=True)
        
        if not user_profile.get('questionnaire_completed', False):
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.warning("Анкета не заполнена. Заполните для получения персонализированных рекомендаций.")
            if st.button("📝 Заполнить анкету", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            with st.form("update_profile_form"):
                st.markdown('<h3 class="section-header">📏 Личные данные</h3>', unsafe_allow_html=True)
                
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
                
                st.markdown('<h3 class="section-header">🎯 Цели</h3>', unsafe_allow_html=True)
                
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
                
                st.markdown('<h3 class="section-header">🏋️‍♀️ Предпочитаемые активности</h3>', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)

# Обработка просмотра деталей программы
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
        st.markdown(f"<h2 class='main-title'>📋 {program_info['name']}</h2>", unsafe_allow_html=True)
        
        level_info = app.levels.get(program_info['level'], {})
        st.markdown(f"<p><strong>Уровень:</strong> {level_info.get('name', 'Начальный')}</p>", unsafe_allow_html=True)
        
        # Получаем все дни тренировок
        workout_days = app.get_all_workout_days(program_id)
        
        if workout_days:
            # Создаем табы для дней
            tabs = st.tabs([f"День {i+1}" for i in range(len(workout_days))])
            
            for i, (tab, day_key) in enumerate(zip(tabs, workout_days)):
                with tab:
                    # Получаем упражнения для дня
                    exercises = app.get_exercises_for_program(program_id, day_key)
                    
                    if exercises:
                        st.markdown(f"<h3 class='section-header'>{exercises.get('title', f'Тренировка {i+1}')}</h3>", unsafe_allow_html=True)
                        
                        # Видео тренировки
                        if 'video_url' in exercises:
                            st.markdown(f"""
                            <div class="modern-card">
                                <h4>🎥 Видео тренировки</h4>
                                <p>{exercises.get('video_description', 'Полная тренировка')}</p>
                                <a href='{exercises['video_url']}' target='_blank' style='color: #3498db; text-decoration: none; font-weight: bold;'>
                                    📺 Смотреть тренировку на YouTube
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Разминка
                        if 'warmup' in exercises:
                            st.markdown(f"<div class='modern-card'><strong>🔥 Разминка:</strong> {exercises['warmup']}</div>", unsafe_allow_html=True)
                        
                        # Упражнения
                        st.markdown('<h4 class="section-header">📋 Упражнения:</h4>', unsafe_allow_html=True)
                        for j, exercise in enumerate(exercises.get('exercises', [])):
                            st.markdown(f"""
                            <div class="nutrition-card" style="margin-bottom: 0.5rem;">
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
                            st.markdown(f"<div class='modern-card'><strong>🧘 Заминка:</strong> {exercises['cooldown']}</div>", unsafe_allow_html=True)
                        
                        # Кнопка для добавления этой тренировки
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            # Используем ключ сессии для определения нажатия кнопки
                            add_workout_key = f"add_workout_{day_key}_{i}"
                            if st.button(f"➕ Добавить тренировку День {i+1}", 
                                       use_container_width=True, 
                                       key=add_workout_key):
                                # Устанавливаем флаги и очищаем форму
                                st.session_state.show_program_details = None
                                st.session_state.selected_day = f"День {i+1}"
                                st.session_state.current_page = "➕ Добавить тренировку"
                                # Также сохраняем информацию о программе для предзаполнения
                                st.session_state.selected_program_for_workout = program_id
                                st.session_state.selected_day_for_workout = f"День {i+1}"
                                # Запускаем перезагрузку
                                st.rerun()
                                
                        with col2:
                            close_key = f"close_program_{i}"
                            if st.button("❌ Закрыть", use_container_width=True, key=close_key):
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
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>💪 <strong>Фитнес Помощник v8.0</strong> | Умный подбор тренировок на основе ваших данных</p>
    <p style='font-size: 0.9rem;'>Ваш персональный тренер для любого вида фитнеса</p>
</div>
""", unsafe_allow_html=True)
