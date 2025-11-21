import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import plotly.express as px
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_title="💪 Фитнес Трекер",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили для красивого оформления
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .exercise-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FitnessApp:
    def __init__(self):
        self.filename = 'data/workouts.csv'
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Создает папку для данных если её нет"""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
    
    def add_workout(self, exercise, weight, reps, sets, notes=''):
        """Добавляет новую тренировку"""
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
            
            if os.path.exists(self.filename):
                existing_df = pd.read_csv(self.filename)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                updated_df = df
                
            updated_df.to_csv(self.filename, index=False)
            return True, "Тренировка успешно сохранена! 💪"
            
        except Exception as e:
            return False, f"Ошибка при сохранении: {e}"
    
    def get_all_workouts(self):
        """Возвращает все тренировки"""
        if os.path.exists(self.filename):
            df = pd.read_csv(self.filename)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date', ascending=False)
        else:
            return pd.DataFrame(columns=['date', 'exercise', 'weight', 'reps', 'sets', 'notes'])
    
    def get_exercise_history(self, exercise_name):
        """Возвращает историю по конкретному упражнению"""
        df = self.get_all_workouts()
        if not df.empty:
            exercise_data = df[df['exercise'] == exercise_name].copy()
            return exercise_data.sort_values('date')
        return pd.DataFrame()
    
    def get_user_exercises(self):
        """Возвращает список всех уникальных упражнений"""
        df = self.get_all_workouts()
        if not df.empty:
            return df['exercise'].unique().tolist()
        return []
    
    def get_statistics(self):
        """Возвращает статистику тренировок"""
        df = self.get_all_workouts()
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
@st.cache_resource
def get_app():
    return FitnessApp()

app = get_app()

# Заголовок приложения
st.markdown('<h1 class="main-header">💪 Фитнес Трекер Pro</h1>', unsafe_allow_html=True)

# Сайдбар для навигации
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3565/3565418.png", width=100)
    st.title("Навигация")
    
    page = st.radio(
        "Выберите раздел:",
        ["📊 Панель управления", "➕ Новая тренировка", "📈 Анализ прогресса", 
         "🤖 Умные прогнозы", "🏆 Достижения", "🔄 Демо-данные"]
    )
    
    st.markdown("---")
    st.markdown("### Статистика")
    
    stats = app.get_statistics()
    if stats:
        st.metric("Всего тренировок", stats['total_workouts'])
        st.metric("Упражнений", stats['unique_exercises'])
        st.metric("Макс. вес", f"{stats['max_weight']:.1f} кг")

# Главная страница - Панель управления
if page == "📊 Панель управления":
    st.markdown('<h2 class="sub-header">📊 Обзор тренировок</h2>', unsafe_allow_html=True)
    
    df = app.get_all_workouts()
    
    if df.empty:
        st.info("🎯 Добро пожаловать в Фитнес Трекер! Начните с добавления первой тренировки.")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3481/3481079.png", width=200)
            st.markdown("""
            <div style='text-align: center;'>
                <h3>Начните свой фитнес-путь!</h3>
                <p>Добавьте первую тренировку чтобы увидеть статистику и прогресс</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Основные метрики
        stats = app.get_statistics()
        
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
        recent_workouts = df.head(10).copy()
        recent_workouts['date'] = recent_workouts['date'].dt.strftime('%d.%m.%Y %H:%M')
        
        # Стилизованное отображение тренировок
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
        
        # График активности
        st.markdown("### Активность по дням")
        if not df.empty:
            df['date_only'] = df['date'].dt.date
            daily_workouts = df.groupby('date_only').size().reset_index()
            daily_workouts.columns = ['date', 'workouts']
            
            fig = px.bar(daily_workouts, x='date', y='workouts', 
                        title='Количество тренировок по дням',
                        color='workouts',
                        color_continuous_scale='blues')
            fig.update_layout(xaxis_title='Дата', yaxis_title='Тренировки')
            st.plotly_chart(fig, use_container_width=True)

# Добавление тренировки
elif page == "➕ Новая тренировка":
    st.markdown('<h2 class="sub-header">➕ Добавить новую тренировку</h2>', unsafe_allow_html=True)
    
    # Переменная для хранения выбранного упражнения
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = ""
    
    # Предустановленные упражнения ДО формы
    st.subheader("Быстрый выбор упражнений")
    preset_cols = st.columns(5)
    preset_exercises = ["Жим лежа", "Приседания", "Становая тяга", "Тяга к поясу", "Жим стоя"]
    
    for i, preset in enumerate(preset_exercises):
        with preset_cols[i]:
            if st.button(preset, use_container_width=True, key=f"preset_{i}"):
                st.session_state.selected_exercise = preset
                st.rerun()
    
    # Форма для ввода данных
    with st.form("workout_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Основные данные")
            exercise = st.text_input(
                "Упражнение 🏋️",
                value=st.session_state.selected_exercise,
                placeholder="Жим лежа, Приседания, Становая тяга...",
                help="Введите название упражнения"
            )
            weight = st.number_input(
                "Вес (кг) ⚖️", 
                min_value=0.0, 
                step=0.5,
                value=50.0,
                help="Рабочий вес в килограммах"
            )
        
        with col2:
            st.subheader("Параметры")
            reps = st.number_input(
                "Количество повторений 🔁", 
                min_value=1, 
                step=1,
                value=8,
                help="Количество повторений в подходе"
            )
            sets = st.number_input(
                "Количество подходов 📊", 
                min_value=1, 
                step=1,
                value=4,
                help="Количество подходов"
            )
        
        notes = st.text_area(
            "Заметки к тренировке 📝", 
            placeholder="Опишите как прошла тренировка, самочувствие, технические моменты...",
            height=100
        )
        
        submitted = st.form_submit_button(
            "💾 Сохранить тренировку", 
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if exercise and weight > 0 and reps > 0 and sets > 0:
                success, message = app.add_workout(exercise, weight, reps, sets, notes)
                if success:
                    st.success(message)
                    st.balloons()
                    
                    # Очищаем выбранное упражнение после успешного сохранения
                    st.session_state.selected_exercise = ""
                    
                    # Показываем сводку сохраненной тренировки
                    st.markdown("### 📋 Сводка тренировки")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Упражнение", exercise)
                    with col2:
                        st.metric("Вес", f"{weight} кг")
                    with col3:
                        st.metric("Повторения", reps)
                    with col4:
                        st.metric("Подходы", sets)
                    
                    # Расчет объема
                    volume = weight * reps * sets
                    st.info(f"**Объем тренировки:** {volume:.0f} кг")
                else:
                    st.error(message)
            else:
                st.error("❌ Заполните все обязательные поля!")

# Анализ прогресса
elif page == "📈 Анализ прогресса":
    st.markdown('<h2 class="sub-header">📈 Детальный анализ прогресса</h2>', unsafe_allow_html=True)
    
    df = app.get_all_workouts()
    
    if df.empty:
        st.warning("📝 Нет данных для анализа. Добавьте несколько тренировок!")
    else:
        exercises = app.get_user_exercises()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_exercise = st.selectbox(
                "Выберите упражнение для анализа:",
                exercises,
                index=0
            )
            
            if selected_exercise:
                exercise_data = app.get_exercise_history(selected_exercise)
                
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
                    
                    # Дополнительная статистика
                    st.markdown("### 📈 Дополнительная информация")
                    latest_workout = exercise_data.iloc[-1]
                    first_workout = exercise_data.iloc[0]
                    
                    st.write(f"**Первая тренировка:** {first_workout['date'].strftime('%d.%m.%Y')} - {first_workout['weight']} кг")
                    st.write(f"**Последняя тренировка:** {latest_workout['date'].strftime('%d.%m.%Y')} - {latest_workout['weight']} кг")
                    st.write(f"**Период тренировок:** {(latest_workout['date'] - first_workout['date']).days} дней")
        
        with col2:
            if selected_exercise and not exercise_data.empty:
                # График прогресса с Matplotlib
                st.markdown("### 📈 График прогресса")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # График 1: Прогресс веса
                ax1.plot(exercise_data['date'], exercise_data['weight'], 'o-', linewidth=2, markersize=6, color='#1f77b4')
                ax1.set_title(f'Прогресс в упражнении: {selected_exercise}', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Вес (кг)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Добавляем аннотации для минимального и максимального веса
                max_idx = exercise_data['weight'].idxmax()
                min_idx = exercise_data['weight'].idxmin()
                ax1.annotate(f'Макс: {exercise_data.loc[max_idx, "weight"]}кг', 
                           xy=(exercise_data.loc[max_idx, 'date'], exercise_data.loc[max_idx, 'weight']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
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
                
                # Таблица с историей тренировок
                st.markdown("### 📋 История тренировок")
                display_data = exercise_data.copy()
                display_data['date'] = display_data['date'].dt.strftime('%d.%m.%Y')
                display_data['volume'] = display_data['weight'] * display_data['reps'] * display_data['sets']
                
                # Показываем только нужные колонки
                display_columns = ['date', 'weight', 'reps', 'sets', 'volume', 'notes']
                st.dataframe(display_data[display_columns], 
                           use_container_width=True,
                           hide_index=True,
                           column_config={
                               'date': 'Дата',
                               'weight': 'Вес (кг)',
                               'reps': 'Повторения',
                               'sets': 'Подходы',
                               'volume': 'Объем',
                               'notes': 'Заметки'
                           })

# Умные прогнозы
elif page == "🤖 Умные прогнозы":
    st.markdown('<h2 class="sub-header">🤖 Умные прогнозы и рекомендации</h2>', unsafe_allow_html=True)
    
    df = app.get_all_workouts()
    
    if len(df) < 3:
        st.warning("""
        ⚠️ Для работы умных прогнозов нужно как минимум 3 тренировки.
        
        **Что делать:**
        1. Добавьте больше тренировок через раздел "➕ Новая тренировка"
        2. Или создайте демо-данные через раздел "🔄 Демо-данные"
        """)
    else:
        exercises = app.get_user_exercises()
        selected_exercise = st.selectbox(
            "Выберите упражнение для анализа:",
            exercises,
            key="ml_exercise"
        )
        
        if selected_exercise:
            exercise_data = app.get_exercise_history(selected_exercise)
            
            if len(exercise_data) >= 3:
                # Подготовка данных для ML
                exercise_data = exercise_data.copy()
                exercise_data = exercise_data.sort_values('date')
                exercise_data['days_passed'] = (exercise_data['date'] - exercise_data['date'].min()).dt.days
                
                # Обучение модели
                X = exercise_data[['days_passed']].values
                y = exercise_data['weight'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Прогноз на будущее
                last_day = exercise_data['days_passed'].max()
                future_days = np.array([
                    [last_day + 7],    # Через 1 неделю
                    [last_day + 14],   # Через 2 недели
                    [last_day + 30],   # Через 1 месяц
                    [last_day + 90]    # Через 3 месяца
                ])
                predictions = model.predict(future_days)
                
                # Текущие показатели
                current_weight = exercise_data['weight'].iloc[-1]
                progress_rate = (current_weight - exercise_data['weight'].iloc[0]) / len(exercise_data)
                
                # Отображение прогнозов
                st.markdown("### 📊 Прогноз прогресса")
                
                col1, col2, col3, col4 = st.columns(4)
                
                time_periods = ["1 неделя", "2 недели", "1 месяц", "3 месяца"]
                deltas = predictions - current_weight
                
                for i, col in enumerate([col1, col2, col3, col4]):
                    with col:
                        st.metric(
                            f"Через {time_periods[i]}",
                            f"{predictions[i]:.1f} кг",
                            delta=f"{deltas[i]:.1f} кг",
                            delta_color="normal" if deltas[i] > 0 else "off"
                        )
                
                # Рекомендации
                st.markdown("### 💡 Персональные рекомендации")
                
                recommendation_col1, recommendation_col2 = st.columns(2)
                
                with recommendation_col1:
                    st.subheader("🎯 Рекомендации по весу")
                    
                    if progress_rate > 0.5:
                        st.success("""
                        **Отличный прогресс! 🎉**
                        - Продолжайте текущую программу
                        - Можно увеличить вес на 2.5-5 кг
                        - Сфокусируйтесь на технике
                        """)
                        recommended_increase = 2.5
                    elif progress_rate > 0.2:
                        st.info("""
                        **Хороший стабильный прогресс 📈**
                        - Увеличивайте вес на 1-2.5 кг
                        - Следите за восстановлением
                        - Чередуйте тяжелые и легкие тренировки
                        """)
                        recommended_increase = 1.0
                    else:
                        st.warning("""
                        **Прогресс медленный ⚡**
                        - Рекомендуется изменить программу
                        - Увеличьте частоту тренировок
                        - Проверьте питание и сон
                        """)
                        recommended_increase = 0.0
                    
                    st.metric(
                        "Рекомендуемый вес",
                        f"{current_weight + recommended_increase} кг",
                        delta=f"+{recommended_increase} кг"
                    )
                
                with recommendation_col2:
                    st.subheader("📈 Статистика прогресса")
                    
                    stats_data = {
                        'Показатель': [
                            'Текущий вес',
                            'Начальный вес', 
                            'Общий прогресс',
                            'Скорость прогресса',
                            'Тренировок выполнено'
                        ],
                        'Значение': [
                            f"{current_weight} кг",
                            f"{exercise_data['weight'].iloc[0]} кг",
                            f"{current_weight - exercise_data['weight'].iloc[0]:.1f} кг",
                            f"{progress_rate:.2f} кг/тренировка",
                            f"{len(exercise_data)}"
                        ]
                    }
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # График с прогнозом
                st.markdown("### 🔮 График прогресса с прогнозом")
                
                # Создаем расширенный dataframe с прогнозами
                future_dates = [
                    exercise_data['date'].max() + timedelta(days=7),
                    exercise_data['date'].max() + timedelta(days=14),
                    exercise_data['date'].max() + timedelta(days=30),
                    exercise_data['date'].max() + timedelta(days=90)
                ]
                
                future_df = pd.DataFrame({
                    'date': future_dates,
                    'weight': predictions,
                    'type': 'Прогноз'
                })
                
                history_df = pd.DataFrame({
                    'date': exercise_data['date'],
                    'weight': exercise_data['weight'],
                    'type': 'История'
                })
                
                combined_df = pd.concat([history_df, future_df])
                
                fig = px.line(combined_df, x='date', y='weight', color='type',
                             title=f'Исторический прогресс и прогноз для {selected_exercise}',
                             color_discrete_map={'История': '#1f77b4', 'Прогноз': '#ff7f0e'})
                
                fig.update_layout(
                    xaxis_title='Дата',
                    yaxis_title='Вес (кг)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Достижения
elif page == "🏆 Достижения":
    st.markdown('<h2 class="sub-header">🏆 Ваши достижения</h2>', unsafe_allow_html=True)
    
    df = app.get_all_workouts()
    
    if df.empty:
        st.info("🎯 Начните тренироваться чтобы получать достижения!")
    else:
        stats = app.get_statistics()
        
        # Система достижений
        achievements = []
        
        # Проверяем достижения
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
        
        # Отображаем достижения
        if achievements:
            st.success(f"🎉 Поздравляем! У вас {len(achievements)} достижений!")
            
            cols = st.columns(3)
            for i, (title, description, color) in enumerate(achievements):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};'>
                            <h4>{title}</h4>
                            <p>{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("""
            **Продолжайте тренироваться!** 🏋️
            
            Достижения появятся когда вы:
            - Выполните 10 тренировок
            - Освоите 5 различных упражнений  
            - Покорите вес в 100 кг
            - Проведете 8+ тренировок за месяц
            """)
        
        # Статистика для новых целей
        st.markdown("### 🎯 Ближайшие цели")
        
        goals_data = []
        
        if stats['total_workouts'] < 10:
            goals_data.append(["10 тренировок", f"{stats['total_workouts']}/10", f"{10 - stats['total_workouts']} осталось"])
        if stats['unique_exercises'] < 5:
            goals_data.append(["5 упражнений", f"{stats['unique_exercises']}/5", f"{5 - stats['unique_exercises']} осталось"])
        if stats['max_weight'] < 100:
            goals_data.append(["Вес 100 кг", f"{stats['max_weight']:.1f}/100", f"{100 - stats['max_weight']:.1f} кг осталось"])
        
        if goals_data:
            goals_df = pd.DataFrame(goals_data, columns=['Цель', 'Прогресс', 'Осталось'])
            st.dataframe(goals_df, use_container_width=True, hide_index=True)
        else:
            st.success("🎊 Все базовые цели достигнуты! Пора ставить новые рекорды!")

# Демо-данные
elif page == "🔄 Демо-данные":
    st.markdown('<h2 class="sub-header">🔄 Создание демо-данных</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Демо-данные** помогут вам протестировать все функции приложения без ввода реальных тренировок.
    
    Будет создана реалистичная история тренировок за последние 2 месяца.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Создать демо-данные", use_container_width=True, type="primary"):
            # Создаем демо-данные с прогрессией
            demo_workouts = []
            base_date = datetime.now() - timedelta(days=60)
            
            # Жим лежа - прогрессия
            for i in range(12):
                date = base_date + timedelta(days=i*5)
                weight = 60 + i * 2.5
                demo_workouts.append((
                    date.strftime('%Y-%m-%d %H:%M:%S'),
                    "Жим лежа",
                    weight,
                    8 if i < 8 else 6,
                    4,
                    f"Тренировка {i+1}, прогресс +{i*2.5}кг"
                ))
            
            # Приседания - прогрессия
            for i in range(10):
                date = base_date + timedelta(days=i*6 + 2)
                weight = 70 + i * 3
                demo_workouts.append((
                    date.strftime('%Y-%m-%d %H:%M:%S'),
                    "Приседания", 
                    weight,
                    6,
                    4,
                    f"Приседания {i+1}, техника улучшается"
                ))
            
            # Становая тяга - прогрессия
            for i in range(8):
                date = base_date + timedelta(days=i*7 + 1)
                weight = 80 + i * 5
                demo_workouts.append((
                    date.strftime('%Y-%m-%d %H:%M:%S'),
                    "Становая тяга",
                    weight,
                    5,
                    3,
                    f"Становая {i+1}, осторожно с техникой"
                ))
            
            # Сохраняем все демо-тренировки
            for workout in demo_workouts:
                app.add_workout(workout[1], workout[2], workout[3], workout[4], workout[5])
            
            st.success("✅ Демо-данные успешно созданы!")
            st.balloons()
            
            st.markdown("""
            ### 📊 Что было создано:
            - **12 тренировок** жима лежа с прогрессией 60 → 87.5 кг
            - **10 тренировок** приседаний с прогрессией 70 → 97 кг  
            - **8 тренировок** становой тяги с прогрессией 80 → 115 кг
            - **Реалистичные даты** за последние 2 месяца
            """)
    
    with col2:
        st.warning("""
        ⚠️ **Внимание!**
        
        При создании демо-данных:
        - Существующие тренировки будут сохранены
        - Добавятся новые демо-тренировки
        - Вы можете удалить их позже через просмотр данных
        """)
        
        if st.button("🗑️ Очистить все данные", type="secondary"):
            if os.path.exists(app.filename):
                os.remove(app.filename)
                st.success("✅ Все данные очищены!")
                st.rerun()

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💪 <strong>Фитнес Трекер Pro v2.0</strong> | Ваш персональный помощник в тренировках</p>
    <p>Отслеживайте прогресс, получайте умные прогнозы и достигайте новых целей!</p>
</div>
""", unsafe_allow_html=True)
