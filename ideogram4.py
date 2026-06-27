import json
import re
import cv2
import numpy as np
import torch

from .qwen3vl_node import CATEGORY_NAME

try:
    import json_repair
except ImportError:
    json_repair = None

class Ideogram4JsonPreviewOnImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json_string": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_preview"
    CATEGORY = CATEGORY_NAME

    def draw_preview(self, image, json_string):
        # 1. Безопасное извлечение и конвертация тензора ComfyUI в OpenCV BGR
        # Убираем батч-размерность, если она есть
        img_tensor = image.cpu().numpy()
        if len(img_tensor.shape) == 4:
            img_tensor = img_tensor[0]
            
        img_bgr = (img_tensor * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        color_map = {
            "красн": (0, 0, 255),    "red": (0, 0, 255),
            "зелен": (0, 255, 0),    "green": (0, 255, 0),
            "син": (255, 0, 0),      "blue": (255, 0, 0),
            "желт": (0, 255, 255),   "yellow": (0, 255, 255),
            "бел": (255, 255, 255),  "white": (255, 255, 255),
            "черн": (0, 0, 0),       "black": (0, 0, 0),
            "розов": (203, 192, 255),"pink": (203, 192, 255),
            "оранж": (0, 165, 255),  "orange": (0, 165, 255),
            "фиолет": (130, 0, 75),  "purple": (130, 0, 75)
        }

        # --- БЛОК ВОССТАНОВЛЕНИЯ JSON ---
        data = None
        error_msg = None
        
        raw_text = json_string.strip()
        if "```" in raw_text:
            blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
            if blocks:
                raw_text = blocks[0].strip()

        if not raw_text.startswith("{") and not raw_text.startswith("["):
            match = re.search(r'(\{[\s\S]*\})', raw_text)
            if match:
                raw_text = match.group(1).strip()

        try:
            if json_repair:
                data = json_repair.loads(raw_text)
            else:
                data = json.loads(raw_text)
        except Exception as e:
            error_msg = f"Parser fail: {str(e)}"

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                pass

        if not isinstance(data, dict):
            err = error_msg if error_msg else "Parsed JSON is not a dictionary."
            cv2.putText(img_bgr, f"JSON REPAIR ERROR: {err[:50]}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return (torch.from_numpy(out_img.astype(np.float32) / 255.0).unsqueeze(0),)
        # --- КОНЕЦ БЛОКА ВОССТАНОВЛЕНИЯ ---

        # Унификация структуры
        if "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]

        objects = []
        if "compositional_deconstruction" in data and isinstance(data["compositional_deconstruction"], dict):
            decomp = data["compositional_deconstruction"]
            objects = decomp.get("elements", [])
        
        if not objects:
            objects = data.get("objects", data.get("elements", data.get("compositional_deconstruction", [])))

        if not isinstance(objects, list):
            objects = [objects]
        
        idx = 1
        # 3. Отрисовка элементов макета
        for obj in objects:
            if not isinstance(obj, dict):
                continue
                
            bbox = obj.get("bbox", obj.get("bounding_box"))
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            
            try:
                ymin, xmin, ymax, xmax = [float(x) for x in bbox]
            except (ValueError, TypeError):
                continue
            
            # Масштабируем под оригинальное разрешение
            start_x = int(xmin * w / 1000)
            start_y = int(ymin * h / 1000)
            end_x = int(xmax * w / 1000)
            end_y = int(ymax * h / 1000)

            prompt_text = obj.get("text") or obj.get("desc") or obj.get("description") or ""
            if not isinstance(prompt_text, str):
                prompt_text = str(prompt_text)
            
            # Дефолтные цвета
            box_color = (0, 255, 255)     # Желтый
            text_color = (255, 255, 255)  # Белый
            
            # Поиск HEX в палитрах
            found_hex = None
            elem_palette = obj.get("color_palette", [])
            if isinstance(elem_palette, list):
                for item in elem_palette:
                    if isinstance(item, str) and re.match(r'#([A-Fa-f0-9]{6})', item):
                        found_hex = item
                        break
            
            if not found_hex:
                hex_match = re.search(r'#([A-Fa-f0-9]{6})', prompt_text)
                if hex_match:
                    found_hex = hex_match.group(0)

            if found_hex:
                hex_val = found_hex.replace("#", "")
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                box_color = (b, g, r)
            else:
                lower_text = prompt_text.lower()
                for keyword, color_bgr in color_map.items():
                    if keyword in lower_text:
                        box_color = color_bgr
                        break

            # ИСПРАВЛЕНО: Поканальное вычисление яркости для исключения деформации матрицы
            brightness = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
            if brightness > 140:
                text_color = (0, 0, 0)

            # Отрисовка
            cv2.rectangle(img_bgr, (start_x, start_y), (end_x, end_y), box_color, 3)

            label = f"#{idx}: {prompt_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            if text_w > (end_x - start_x - 10):
                slice_len = max(5, int(len(label) * (end_x - start_x) / text_w) - 4)
                label = label[:slice_len] + "..."
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            text_box_y = max(start_y, text_h + 12)

            cv2.rectangle(img_bgr, (start_x, text_box_y - text_h - 12), (start_x + text_w + 10, text_box_y), box_color, -1)
            cv2.putText(img_bgr, label, (start_x + 5, text_box_y - 6), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            idx += 1

        # 4. Переводим обратно в тензор ComfyUI с правильной формой [1, H, W, C]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (out_tensor,)


class Ideogram4JsonSwapCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "swap_coordinates"
    CATEGORY = CATEGORY_NAME

    def swap_coordinates(self, json_string):
        # 1. БЛОК ВОССТАНОВЛЕНИЯ И ОЧИСТКИ JSON (Аналогично первой ноде)
        data = None
        
        raw_text = json_string.strip()
        if "```" in raw_text:
            blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
            if blocks:
                raw_text = blocks[0].strip()

        if not raw_text.startswith("{") and not raw_text.startswith("["):
            match = re.search(r'(\{[\s\S]*\})', raw_text)
            if match:
                raw_text = match.group(1).strip()

        try:
            if json_repair:
                data = json_repair.loads(raw_text)
            else:
                data = json.loads(raw_text)
        except Exception as e:
            return (f'{{"error": "Parser fail: {str(e)}"}}',)

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                pass

        if not isinstance(data, dict):
            return ('{"error": "Parsed JSON is not a valid dictionary."}',)

        # 2. ПОИСК И СВАП КООРДИНАТ [x_min, y_min, x_max, y_max] -> [y_min, x_min, y_max, x_max]
        # Рекурсивная функция для глубокого поиска списков координат по всему словарю
        def process_node(node):
            if isinstance(node, dict):
                # Проверяем наличие ключей координат
                for key in ["bbox", "bounding_box"]:
                    if key in node and isinstance(node[key], list) and len(node[key]) == 4:
                        try:
                            # Извлекаем текущие координаты, сгенерированные Qwen: [xmin, ymin, xmax, ymax]
                            xmin, ymin, xmax, ymax = [float(x) for x in node[key]]
                            
                            # Превращаем в формат Ideogram 4: [ymin, xmin, ymax, xmax]
                            # Округляем до int, так как Ideogram оперирует целыми числами от 0 до 1000
                            node[key] = [int(ymin), int(xmin), int(ymax), int(xmax)]
                        except (ValueError, TypeError):
                            pass # Пропускаем, если внутри не числа
                
                # Обходим все вложенные словари
                for k, v in node.items():
                    process_node(v)
                    
            elif isinstance(node, list):
                # Обходим списки (например, список elements/objects)
                for item in node:
                    process_node(item)

        # Запускаем трансформацию структуры координат
        process_node(data)

        # 3. Сериализуем исправленный словарь обратно в чистую JSON-строку
        # indent=2 делает строку красивой и читаемой в ComfyUI логах и текстовых окнах
        fixed_json_string = json.dumps(data, ensure_ascii=False, indent=2)
        
        return (fixed_json_string,)

# ------------------------------------------------------------------
# Маппинги для ComfyUI
# ------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Ideogram4JsonPreviewOnImage": Ideogram4JsonPreviewOnImage,
    "Ideogram4JsonSwapCoordinates": Ideogram4JsonSwapCoordinates
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4JsonPreviewOnImage": "📐 Ideogram 4 JSON Repair & Preview",
    "Ideogram4JsonSwapCoordinates": "🔄 Ideogram 4 JSON Swap XY Coordinates"
}