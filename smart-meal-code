rom telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes, filters
)
import os
import asyncio
from functools import partial
from google.cloud import vision
import requests
from ultralytics import YOLO
import cv2

TOKEN = "---"  # Telegram Bot token
SPOONACULAR_API_KEY = "------"  # Spoonacular API key

PHOTO_WAIT = 1
RECIPES_PER_PAGE = 4

# ------------------------------
# Initialize clients
# ------------------------------
client = vision.ImageAnnotatorClient()
yolo_model = YOLO("yolov8n.pt")

# ------------------------------
# Load ingredients whitelist
# ------------------------------
INGREDIENTS_FILE = r"C:\Users\User\Desktop\bot\ingredients.txt"
with open(INGREDIENTS_FILE, "r", encoding="utf-8") as f:
    ALLOWED_INGREDIENTS = set(line.strip().lower() for line in f if line.strip())
YOLO_FOOD_WHITELIST = ALLOWED_INGREDIENTS

# ------------------------------
# In-memory storage
# ------------------------------
USER_RECIPES = {}          # {user_id: {"all": [], "saved": []}}
USER_CURRENT_PAGE = {}     # {user_id: page_index}

# ------------------------------
# Detection Functions
# ------------------------------
def detect_ingredients_vision(photo_path):
    with open(photo_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = [label.description.lower() for label in response.label_annotations]
    return [label for label in labels if label in ALLOWED_INGREDIENTS]

def crop_image_grid(image_path, rows=3, cols=3):
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    crops = []
    crop_h = h // rows
    crop_w = w // cols
    for i in range(rows):
        for j in range(cols):
            y1 = i * crop_h
            x1 = j * crop_w
            y2 = h if i == rows - 1 else (i + 1) * crop_h
            x2 = w if j == cols - 1 else (j + 1) * crop_w
            crops.append(img[y1:y2, x1:x2])
    return crops

def detect_ingredients_yolo_crops(photo_path, rows=3, cols=3):
    crops = crop_image_grid(photo_path, rows, cols)
    detected = set()
    for crop in crops:
        results = yolo_model(crop)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls].lower()
                conf = float(box.conf[0])
                if conf > 0.5 and label in YOLO_FOOD_WHITELIST:
                    detected.add(label)
    return list(detected)

def detect_ingredients_combined(photo_path):
    vision_items = set(detect_ingredients_vision(photo_path))
    yolo_items = set(detect_ingredients_yolo_crops(photo_path))
    return list(vision_items.union(yolo_items))

# ------------------------------
# Recipe Search
# ------------------------------
def find_recipes(ingredients):
    if not ingredients:
        return []
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ",".join(ingredients),
        "number": 20,
        "ranking": 1,
        "ignorePantry": True,
        "apiKey": SPOONACULAR_API_KEY,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.RequestException:
        return []
    data = response.json()
    recipes = []
    for item in data:
        title = item.get("title")
        recipe_id = item.get("id")
        link = f"https://spoonacular.com/recipes/{title.replace(' ', '-')}-{recipe_id}"
        recipes.append(f"{title} - {link}")
    return recipes

# ------------------------------
# Pagination & Buttons
# ------------------------------
async def send_recipe_page(update, context, user_id, edit=False):
    data = USER_RECIPES.get(user_id)
    if not data:
        return
    all_recipes = data["all"]
    page = USER_CURRENT_PAGE.get(user_id, 0)
    start = page * RECIPES_PER_PAGE
    end = start + RECIPES_PER_PAGE
    page_recipes = all_recipes[start:end]
    if not page_recipes:
        return

    keyboard = []
    row = []
    for i, recipe in enumerate(page_recipes):
        number = start + i + 1
        row.append(InlineKeyboardButton(str(number), callback_data=f"save_{number-1}"))
    keyboard.append(row)

    arrows = []
    if start > 0:
        arrows.append(InlineKeyboardButton("⬅️", callback_data="prev"))
    if end < len(all_recipes):
        arrows.append(InlineKeyboardButton("➡️", callback_data="next"))
    if arrows:
        keyboard.append(arrows)

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "Выбери номер рецепта чтобы сохранить:\n\n"
    for i, recipe in enumerate(page_recipes):
        number = start + i + 1
        text += f"{number}. {recipe}\n\n"

    if edit:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup)

# ------------------------------
# Button handler
# ------------------------------
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    data = USER_RECIPES.get(user_id)

    # Навигация
    if query.data == "next":
        USER_CURRENT_PAGE[user_id] += 1
        await send_recipe_page(update, context, user_id, edit=True)
    elif query.data == "prev":
        USER_CURRENT_PAGE[user_id] -= 1
        await send_recipe_page(update, context, user_id, edit=True)

    # Сохранение рецепта
    elif query.data.startswith("save_"):
        index = int(query.data.split("_")[1])
        recipe = data["all"][index]
        data.setdefault("saved", []).append(recipe)
        await query.answer(f"Рецепт под номером {index+1} сохранен! Напиши /myrecipes чтобы посмотреть сохраненные рецепты!", show_alert=True)

    # Удаление рецепта
    elif query.data == "delete_recipe":
        saved = data.get("saved", [])
        if not saved:
            await query.answer("Нет рецептов для удаления.", show_alert=True)
            return
        USER_CURRENT_PAGE[user_id] = 0
        await show_delete_page(update, user_id, edit=True)

    elif query.data.startswith("del_"):
        idx = int(query.data.split("_")[1])
        saved = data.get("saved", [])
        if 0 <= idx < len(saved):
            deleted = saved.pop(idx)
            await query.answer(f"Удалён рецепт: {deleted}", show_alert=True)
            # Обновляем кнопки для удаления
            await show_delete_page(update, user_id, edit=True)

    elif query.data == "del_next":
        USER_CURRENT_PAGE[user_id] += 1
        await show_delete_page(update, user_id, edit=True)
    elif query.data == "del_prev":
        USER_CURRENT_PAGE[user_id] -= 1
        await show_delete_page(update, user_id, edit=True)

# ------------------------------
# Delete recipe pagination
# ------------------------------
async def show_delete_page(update, user_id, edit=False):
    data = USER_RECIPES.get(user_id)
    saved = data.get("saved", [])
    if not saved:
        await update.callback_query.edit_message_text("У тебя нет сохранённых рецептов 😔")
        return

    page = USER_CURRENT_PAGE.get(user_id, 0)
    start = page * RECIPES_PER_PAGE
    end = start + RECIPES_PER_PAGE
    page_recipes = saved[start:end]

    keyboard = []
    row = []
    for i, _ in enumerate(page_recipes):
        number = start + i
        row.append(InlineKeyboardButton(str(i+1), callback_data=f"del_{number}"))
    keyboard.append(row)

    arrows = []
    if start > 0:
        arrows.append(InlineKeyboardButton("⬅️", callback_data="del_prev"))
    if end < len(saved):
        arrows.append(InlineKeyboardButton("➡️", callback_data="del_next"))
    if arrows:
        keyboard.append(arrows)

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "Какой рецепт удалить?\n\n"
    for i, recipe in enumerate(page_recipes):
        number = start + i + 1
        text += f"{number}. {recipe}\n\n"

    if edit:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(text, reply_markup=reply_markup)

# ------------------------------
# Handlers
# ------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я - smart-meal_AI, бот, который найдёт рецепты из остатков в холодильнике.\n"
        "Используй /picture чтобы отправить фото еды.\n"
        "Команда /myrecipes покажет сохранённые рецепты."
    )

async def picture_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправь фото еды.")
    return PHOTO_WAIT

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    temp_file = f"temp_{update.effective_user.id}.jpg"
    await photo_file.download_to_drive(temp_file)
    await update.message.reply_text("Обрабатываю фото...")

    loop = asyncio.get_running_loop()
    ingredients = await loop.run_in_executor(None, partial(detect_ingredients_combined, temp_file))
    os.remove(temp_file)

    if not ingredients:
        await update.message.reply_text("Не удалось распознать ингредиенты 😔")
        return PHOTO_WAIT

    await update.message.reply_text("Я нашёл следующие ингредиенты:\n- " + "\n- ".join(ingredients))

    recipes = await loop.run_in_executor(None, partial(find_recipes, ingredients))
    if not recipes:
        await update.message.reply_text("Не удалось найти рецепты 😔")
        return PHOTO_WAIT

    USER_RECIPES[update.effective_user.id] = {
        "all": recipes,
        "saved": USER_RECIPES.get(update.effective_user.id, {}).get("saved", [])
    }
    USER_CURRENT_PAGE[update.effective_user.id] = 0
    await send_recipe_page(update, context, update.effective_user.id)
    await update.message.reply_text(
        "Отправь ещё одно фото если хочешь найти больше рецептов!"
    )
    return PHOTO_WAIT

async def myrecipes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    data = USER_RECIPES.get(user_id)
    if not data or not data.get("saved"):
        await update.message.reply_text("У тебя пока нет сохранённых рецептов.")
        return
    saved = list(reversed(data["saved"]))
    text = "Твои сохранённые рецепты:\n\n"
    for recipe in saved:
        text += f"- {recipe}\n\n"
    keyboard = [[InlineKeyboardButton("Удалить рецепт", callback_data="delete_recipe")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот остановлен.")
    return ConversationHandler.END

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("picture", picture_command)],
        states={PHOTO_WAIT: [MessageHandler(filters.PHOTO, handle_photo)]},
        fallbacks=[CommandHandler("stop", stop_command)],
    )
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("myrecipes", myrecipes_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(conv_handler)
    print("Бот запущен...")
    app.run_polling()
