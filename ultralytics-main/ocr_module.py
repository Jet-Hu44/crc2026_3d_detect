"""
轻量OCR模块 — 用于识别未知物品(W类)表面文字以区分小类

依赖: pytesseract, opencv-python (已在环境中)
系统依赖: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
"""

import cv2
import numpy as np

# 延迟导入，避免未安装时崩溃
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("[OCR] pytesseract未安装，OCR功能不可用。安装: pip install pytesseract")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[OCR] Pillow未安装。安装: pip install Pillow")


class LightweightOCR:
    """轻量OCR识别器，在香橙派ARM CPU上运行，无GPU依赖

    使用Tesseract作为OCR引擎，支持中英文混合识别。
    比赛前根据QQ群公布的小类更新 subclass_keywords 映射表。
    """

    # ========== 赛前根据公布的小类修改此映射表 ==========
    # 格式: {关键词: 子类编号}
    # 例如赛前公布未知物品小类为"数学类书籍W01"、"英语类书籍W02"
    SUBCLASS_KEYWORDS = {
        # --- 书籍类 ---
        '数学': 'W01',
        '高等数学': 'W01',
        '线性代数': 'W01',
        '概率论': 'W01',
        '英语': 'W02',
        '大学英语': 'W02',
        '物理': 'W03',
        '大学物理': 'W03',
        '化学': 'W04',
        # --- 饮料/食品包装 ---
        '可乐': 'W10',
        '雪碧': 'W10',
        '芬达': 'W10',
        '农夫': 'W11',
        '怡宝': 'W11',
        # --- 其他 ---
        '笔记本': 'W20',
        '文具': 'W20',
    }

    def __init__(self, lang='chi_sim+eng', conf_threshold=30):
        """
        Args:
            lang: Tesseract语言包，默认 chi_sim+eng (中英混合)
            conf_threshold: OCR置信度阈值(0-100)，低于此值的识别结果丢弃
        """
        self.lang = lang
        self.conf_threshold = conf_threshold
        self.available = HAS_TESSERACT and HAS_PIL

        if not self.available:
            print("[OCR] 未就绪，W类物品将使用YOLO外观分类（不读取表面文字）")
        else:
            # 验证语言包
            try:
                langs = pytesseract.get_languages()
                required = [l.strip() for l in lang.split('+')]
                missing = [l for l in required if l not in langs]
                if missing:
                    print(f"[OCR] 警告: 缺少语言包 {missing}，请安装:")
                    for m in missing:
                        print(f"  sudo apt-get install tesseract-ocr-{m}")
                else:
                    print(f"[OCR] 就绪，语言: {lang}")
            except Exception as e:
                print(f"[OCR] Tesseract初始化警告: {e}")

    def preprocess(self, crop_bgr):
        """图像预处理: 灰度→降噪→二值化，提升OCR准确率"""
        # 转灰度
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE增强对比度（处理光照不均）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 去噪
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # OTSU自适应二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def recognize(self, image, bbox):
        """对检测框内的区域进行OCR识别

        Args:
            image: BGR格式的完整帧 (numpy array)
            bbox: (x1, y1, x2, y2) 检测框坐标

        Returns:
            recognized_text: 识别到的文字，失败返回空字符串
        """
        if not self.available:
            return ''

        x1, y1, x2, y2 = bbox
        # 边界安全检查 + 略微扩展ROI（确保文字边缘不被截断）
        h, w = image.shape[:2]
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return ''  # ROI太小，无法OCR

        try:
            # 预处理
            processed = self.preprocess(crop)

            # Tesseract OCR
            # --oem 3: LSTM+Legacy混合引擎
            # --psm 6: 假设为均匀的文本块
            config = f'--oem 3 --psm 6 -l {self.lang}'
            pil_img = Image.fromarray(processed)
            ocr_data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)

            # 提取高置信度的文本
            texts = []
            for i, text in enumerate(ocr_data['text']):
                conf = int(ocr_data['conf'][i])
                text = text.strip()
                if text and conf >= self.conf_threshold and len(text) >= 2:
                    texts.append(text)

            return ''.join(texts)

        except Exception as e:
            print(f"[OCR] 识别异常: {e}")
            return ''

    def classify_subclass(self, recognized_text):
        """根据OCR识别的文字，匹配子类编号

        Args:
            recognized_text: OCR识别到的完整文本

        Returns:
            subclass_id: 子类编号 (如 'W01')，未匹配到返回 None
        """
        if not recognized_text:
            return None

        # 按关键词长度降序匹配（优先匹配长关键词，避免"高等数学"被"数学"短路）
        sorted_keywords = sorted(self.SUBCLASS_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in recognized_text:
                subclass_id = self.SUBCLASS_KEYWORDS[keyword]
                print(f"[OCR] 匹配: '{recognized_text}' → 关键词'{keyword}' → {subclass_id}")
                return subclass_id

        print(f"[OCR] 未匹配关键词: '{recognized_text}'")
        return None

    def update_keywords(self, new_mapping):
        """赛前更新关键词映射表

        Args:
            new_mapping: dict, {关键词: 子类编号}
        """
        self.SUBCLASS_KEYWORDS.update(new_mapping)
        print(f"[OCR] 关键词表已更新，当前 {len(self.SUBCLASS_KEYWORDS)} 条")


# ========== 简易OCR备选方案（纯OpenCV，无外部依赖） ==========

class SimpleTemplateOCR:
    """备选方案: 基于模板匹配的简易文字识别

    如果Tesseract因故无法安装，可用此方案应急。
    仅能识别预定义模板中的固定文字，灵活度低但可靠。
    """

    def __init__(self, templates_dir=None):
        self.templates = {}  # {template_name: template_image}
        self.available = False  # 需赛前准备模板图片

    def add_template(self, name, image_path):
        template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            self.templates[name] = template
            self.available = True

    def match(self, crop_bgr, threshold=0.6):
        if not self.available:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        best_match, best_score = None, 0
        for name, template in self.templates.items():
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score and max_val >= threshold:
                best_score = max_val
                best_match = name
        return best_match
