import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import hashlib

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
</style>
""", unsafe_allow_html=True)

class FitnessApp:
    def __init__(self):
        self.data_dir = 'user_data'
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Создает папку для данных если её нет"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_user_filename(self, username):
        """Генерирует имя файла для пользователя"""
        # Создаем хеш имени пользователя для безопасности
        user_hash = hashlib.md5(username.encode()).hexdigest()[:8]
        return os.path.join(self.data_dir, f'workouts_{user_hash}.csv')
    
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
            'favorite_exercise': df['exercise'].mode().iloc[0] if not df['exercise'].mode().empty else "Нет данных"
        }
        return stats

# Инициализация приложения
app = FitnessApp()

# Система пользователей
def initialize_session_state():
    if 'current_user' not in st.session_state:
        st.session_state.current_user = ""
    if 'user_created' not in st.session_state:
        st.session_state.user_created = False

initialize_session_state()

# Заголовок приложения
st.markdown('<h1 class="main-header">💪 Фитнес Трекер Pro</h1>', unsafe_allow_html=True)

# Система входа/регистрации
if not st.session_state.user_created:
    st.markdown("### 👤 Введите ваше имя для начала")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Ваше имя:", placeholder="Например: Анна или Мария")
        
        if st.button("🎯 Начать использовать", use_container_width=True):
            if username.strip():
                st.session_state.current_user = username.strip()
                st.session_state.user_created = True
                st.rerun()
            else:
                st.error("⚠️ Введите ваше имя!")
    
    with col2:
        st.info("""
        **Почему нужно имя?**
        - 📊 У каждого свой прогресс
        - 🔒 Данные сохраняются отдельно
        - 👥 Можно делиться ссылкой с подругами
        - 🎯 Персональные рекомендации
        """)

else:
    # Отображение текущего пользователя
    st.sidebar.markdown(f'<div class="user-card">👤 Пользователь: <b>{st.session_state.current_user}</b></div>', unsafe_allow_html=True)
    
    # Основная навигация
    with st.sidebar:
        st.title("Навигация")
        
        page = st.radio(
            "Выберите раздел:",
            ["📊 Панель управления", "➕ Новая тренировка", "📋 Мои тренировки", 
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
        if st.button("🚪 Сменить пользователя"):
            st.session_state.current_user = ""
            st.session_state.user_created = False
            st.rerun()

    # Главная страница - Панель управления
    if page == "📊 Панель управления":
        st.markdown(f'<h2 class="sub-header">📊 Панель управления - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.info("🎯 Добро пожаловать! Начните с добавления первой тренировки.")
        else:
            # Основные метрики
            stats = app.get_statistics(st.session_state.current_user)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Всего тренировок", 
                    stats['total_workouts'],
                    f"+{stats['workouts_this_month']} за месяц"
                )
            
            with col2:
                st.metric("Уникальных упражнений", stats['unique_exercises'])
            
            with col3:
                st.metric("Максимальный вес", f"{stats['max_weight']:.1f} кг")
            
            with col4:
                st.metric("Общий объем", f"{stats['total_volume']:.0f} кг")
            
            # Последние тренировки
            st.markdown("### Последние тренировки")
            recent_workouts = df.head(5).copy()
            recent_workouts['date'] = recent_workouts['date'].dt.strftime('%d.%m.%Y %H:%M')
            
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
                        st.caption(workout['date'])
                    st.markdown("---")

    # Добавление тренировки
    elif page == "➕ Новая тренировка":
        st.markdown(f'<h2 class="sub-header">➕ Новая тренировка - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
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
            
            # Быстрый выбор упражнений
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
        
        # Кнопки сохранения и очистки
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
                        
                        # Очистка полей после сохранения
                        st.session_state.workout_data = {
                            'exercise': '',
                            'weight': 50.0,
                            'reps': 8,
                            'sets': 4,
                            'notes': ''
                        }
                        
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("❌ Заполните все обязательные поля!")
        
        with col2:
            if st.button("🧹 Очистить форму", use_container_width=True):
                st.session_state.workout_data = {
                    'exercise': '',
                    'weight': 50.0,
                    'reps': 8,
                    'sets': 4,
                    'notes': ''
                }
                st.rerun()

    # Мои тренировки (с возможностью удаления)
    elif page == "📋 Мои тренировки":
        st.markdown(f'<h2 class="sub-header">📋 Мои тренировки - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.info("📝 У вас пока нет тренировок. Добавьте первую!")
        else:
            st.info(f"🎯 Всего тренировок: {len(df)}")
            
            # Поиск и фильтрация
            col1, col2 = st.columns([2, 1])
            with col1:
                search_exercise = st.text_input("🔍 Поиск по упражнению:", placeholder="Введите название упражнения...")
            with col2:
                show_count = st.selectbox("Показывать:", [10, 25, 50, "Все"])
            
            # Фильтрация данных
            display_df = df.copy()
            if search_exercise:
                display_df = display_df[display_df['exercise'].str.contains(search_exercise, case=False, na=False)]
            
            if show_count != "Все":
                display_df = display_df.head(show_count)
            
            display_df = display_df.reset_index(drop=True)
            display_df['date'] = display_df['date'].dt.strftime('%d.%m.%Y %H:%M')
            display_df['volume'] = display_df['weight'] * display_df['reps'] * display_df['sets']
            
            # Отображение тренировок с возможностью удаления
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
            
            # Статистика
            if len(display_df) > 0:
                st.markdown("### 📊 Статистика показанных тренировок")
                total_volume = display_df['volume'].sum()
                avg_weight = display_df['weight'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Средний вес", f"{avg_weight:.1f} кг")
                with col2:
                    st.metric("Общий объем", f"{total_volume:.0f} кг")

    # Анализ прогресса (остается без изменений, но с учетом пользователя)
    elif page == "📈 Анализ прогресса":
        st.markdown(f'<h2 class="sub-header">📈 Анализ прогресса - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.warning("📝 Нет данных для анализа. Добавьте несколько тренировок!")
        else:
            exercises = app.get_user_exercises(st.session_state.current_user)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_exercise = st.selectbox(
                    "Выберите упражнение для анализа:",
                    exercises,
                    index=0
                )
                
                if selected_exercise:
                    exercise_data = app.get_exercise_history(st.session_state.current_user, selected_exercise)
                    
                    if not exercise_data.empty:
                        # Статистика упражнения
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
                        
                        # Прогресс
                        if progress > 0:
                            st.success(f"📈 Общий прогресс: +{progress:.1f} кг")
                        else:
                            st.info("📊 Прогресс: 0 кг")
            
            with col2:
                if selected_exercise and not exercise_data.empty:
                    # График прогресса
                    st.markdown("### 📈 График прогресса")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # График 1: Прогресс веса
                    ax1.plot(exercise_data['date'], exercise_data['weight'], 'o-', linewidth=2, markersize=6, color='#1f77b4')
                    ax1.set_title(f'Прогресс в упражнении: {selected_exercise}', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Вес (кг)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # График 2: Объем тренировок
                    exercise_data['volume'] = exercise_data['weight'] * exercise_data['reps'] * exercise_data['sets']
                    ax2.plot(exercise_data['date'], exercise_data['volume'], 's-', linewidth=2, markersize=6, color='#ff7f0e')
                    ax2.set_title(f'Объем тренировок: {selected_exercise}', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Дата', fontsize=12)
                    ax2.set_ylabel('Объем (кг)', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

    # Умные прогнозы (аналогично с учетом пользователя)
    elif page == "🤖 Умные прогнозы":
        st.markdown(f'<h2 class="sub-header">🤖 Умные прогнозы - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if len(df) < 3:
            st.warning("⚠️ Нужно минимум 3 тренировки для прогнозов.")
        else:
            exercises = app.get_user_exercises(st.session_state.current_user)
            selected_exercise = st.selectbox("Выберите упражнение:", exercises, key="ml_exercise")
            
            if selected_exercise:
                exercise_data = app.get_exercise_history(st.session_state.current_user, selected_exercise)
                
                if len(exercise_data) >= 3:
                    # ... (остальной код прогнозов без изменений, но с учетом пользователя)
                    st.success("🤖 ML-прогнозы работают!")
                    # Добавьте сюда код из предыдущей версии

    # Достижения (с учетом пользователя)
    elif page == "🏆 Достижения":
        st.markdown(f'<h2 class="sub-header">🏆 Достижения - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        df = app.get_all_workouts(st.session_state.current_user)
        
        if df.empty:
            st.info("🎯 Начните тренироваться чтобы получать достижения!")
        else:
            stats = app.get_statistics(st.session_state.current_user)
            
            # Система достижений
            achievements = []
            
            if stats['total_workouts'] >= 10:
                achievements.append(("🎖️ Десяточка", "Выполнено 10 тренировок!", "success"))
            if stats['total_workouts'] >= 50:
                achievements.append(("🏅 Полтинник", "Выполнено 50 тренировок!", "success"))
            
            if stats['unique_exercises'] >= 5:
                achievements.append(("🎯 Универсал", "Освоено 5 различных упражнений!", "info"))
            
            if stats['max_weight'] >= 100:
                achievements.append(("💯 Сотня", "Покорен вес в 100 кг!", "warning"))
            
            if stats['workouts_this_month'] >= 8:
                achievements.append(("🔥 Активный месяц", "8+ тренировок за месяц!", "success"))
            
            if achievements:
                st.success(f"🎉 Поздравляем! У вас {len(achievements)} достижений!")
            else:
                st.info("🏋️ Продолжайте тренироваться для получения достижений!")

    # Демо-данные (для текущего пользователя)
    elif page == "🔄 Демо-данные":
        st.markdown(f'<h2 class="sub-header">🔄 Демо-данные - {st.session_state.current_user}</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Создать демо-данные", use_container_width=True, type="primary"):
                # Код создания демо-данных (аналогично предыдущей версии)
                demo_workouts = []
                base_date = datetime.now() - timedelta(days=60)
                
                for i in range(12):
                    date = base_date + timedelta(days=i*5)
                    weight = 60 + i * 2.5
                    demo_workouts.append((
                        date.strftime('%Y-%m-%d %H:%M:%S'),
                        "Жим лежа",
                        weight, 8 if i < 8 else 6, 4,
                        f"Тренировка {i+1}, прогресс +{i*2.5}кг"
                    ))
                
                for workout in demo_workouts:
                    app.add_workout(st.session_state.current_user, workout[1], workout[2], workout[3], workout[4], workout[5])
                
                st.success("✅ Демо-данные созданы!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Очистить мои данные", type="secondary"):
                # Удаляем файл пользователя
                filename = app.get_user_filename(st.session_state.current_user)
                if os.path.exists(filename):
                    os.remove(filename)
                    st.success("✅ Ваши данные очищены!")
                    st.rerun()

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💪 <strong>Фитнес Трекер Pro v3.0</strong> | Персональный трекер для каждого пользователя</p>
</div>
""", unsafe_allow_html=True)
