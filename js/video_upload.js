import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "VideoFragmentLivePreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SimpleLoadVideoFragment") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                // ---- ИНИЦИАЛИЗАЦИЯ СВОЙСТВ ----
                if (!this.properties) this.properties = {};
                if (this.properties.preview_max_frames === undefined) {
                    this.properties.preview_max_frames = 1000; // По умолчанию 1000 кадров
                }
                if (this.properties.preview_fps === undefined) {
                    this.properties.preview_fps = 24; // 0 = использовать target_fps
                }
                if (this.properties.preview_longer_size === undefined) {
                    this.properties.preview_longer_size = 400; // По умолчанию 400px по длинной стороне
                }
                if (this.properties.preview_megapixels === undefined) {
                    this.properties.preview_megapixels = 0.0; // По умолчанию не задано
                }
                if (this.properties.preview_jpeg_quality === undefined) {
                    this.properties.preview_jpeg_quality = 70; // По умолчанию 70 
                }

                this._isPreviewPlaying = false;
                this._suppressCropCallback = false;
                this._total_duration = null;         // общая длительность видео
                this._isAdjusting = false;           // флаг для предотвращения рекурсии при корректировке
                this._currentVideoPath = null;       // ОТСЛЕЖИВАНИЕ ТЕКУЩЕГО ВИДЕО
                this._previewAbortController = null; // КОНТРОЛЛЕР ДЛЯ ОТМЕНЫ ЗАПРОСОВ

                // ---- ПОЛУЧАЕМ ВИДЖЕТЫ ----
                const pathWidget = this.widgets.find(w => w.name === "video_path");
                const timestampWidget = this.widgets.find(w => w.name === "timestamp");
                const durationWidget = this.widgets.find(w => w.name === "duration_sec");
                const fpsWidget = this.widgets.find(w => w.name === "target_fps");
                this._timestampWidget = timestampWidget;

                // ---- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВРЕМЕНИ ----
                this._secondsToTimestamp = (sec) => {
                    sec = Math.max(0, sec);
                    const hours = Math.floor(sec / 3600);
                    const minutes = Math.floor((sec % 3600) / 60);
                    const seconds = sec % 60;
                    const secStr = seconds % 1 === 0 ? String(seconds).padStart(2, '0') : seconds.toFixed(2).padStart(5, '0');
                    return hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${secStr}` : `${minutes}:${secStr}`;
                };

                this._timestampToSeconds = (ts) => {
                    if (!ts || ts === "0:00:00") return 0;
                    ts = ts.trim();
                    let ms = 0, mainPart = ts;
                    if (ts.includes('.')) {
                        const [main, msPart] = ts.split('.', 2);
                        ms = parseFloat('0.' + msPart);
                        mainPart = main;
                    }
                    const parts = mainPart.split(':');
                    if (parts.length === 1) return parseFloat(parts[0]) + ms;
                    if (parts.length === 2) return parseInt(parts[0]) * 60 + parseFloat(parts[1]) + ms;
                    if (parts.length === 3) return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2]) + ms;
                    return 0;
                };

                // ---- КНОПКА BROWSE ----
                if (pathWidget) {
                    this.addWidget("button", " Browse...", "browse", async () => {
                        try {
                            const resp = await fetch("/video_fragment/open_file_dialog");
                            const data = await resp.json();
                            if (data.path) {
                                pathWidget.value = data.path;
                                pathWidget.callback?.(data.path);
                            }
                        } catch (e) {
                            console.error("[Load Video Fragment] File dialog error:", e);
                        }
                    });
                }

                // ---- УДАЛЕНИЕ НОДЫ ----
                const onRemoved = nodeType.prototype.onRemoved;
                nodeType.prototype.onRemoved = function() {
                    if (this._previewInterval) clearInterval(this._previewInterval);
                    if (this._animationFrameId) cancelAnimationFrame(this._animationFrameId);
                    
                    // ОЧИСТКА АУДИО
                    if (this._audioEl) {
                        this._audioEl.pause();
                        if (this._audioEl.src) {
                            URL.revokeObjectURL(this._audioEl.src);
                            this._audioEl.src = '';
                        }
                        this._audioEl.remove();
                        this._audioEl = null;
                    }

                    onRemoved?.apply(this, arguments);
                };

                // ---- СОЗДАНИЕ UI ВИДЖЕТОВ ----
                this.timelineWidget = createTimelineWidget(this);
                this.seekControlsWidget = createSeekControlsWidget(this);
                this._preview = createPreviewWidget(this);

                // ---- ФУНКЦИЯ СКРЫТИЯ/ПОКАЗА ВИДЖЕТОВ CROP ----
                this.toggleCropWidgets = (visible) => {
                    const cropWidgetNames = ['crop_x1', 'crop_y1', 'crop_x2', 'crop_y2'];
                    cropWidgetNames.forEach(name => {
                        const widget = this.widgets.find(w => w.name === name);
                        if (widget) {
                            widget.hidden = !visible;
                        }
                    });
                    // Пересчитываем размер ноды
                    requestAnimationFrame(() => {
                        const newSize = this.computeSize();
                        this.setSize([this.size[0], newSize[1]]);
                        this.setDirtyCanvas(true, true);
                    });
                };

                // ---- ФУНКЦИЯ СКРЫТИЯ/ПОКАЗА ВИДЖЕТОВ RESIZE ----
                this.toggleResizeWidgets = (visible) => {
                    const resizeWidgetNames = ['longer_size', 'megapixels', 'size_multiple'];
                    resizeWidgetNames.forEach(name => {
                        const widget = this.widgets.find(w => w.name === name);
                        if (widget) {
                            widget.hidden = !visible;
                        }
                    });
                    // Пересчитываем размер ноды
                    requestAnimationFrame(() => {
                        const newSize = this.computeSize();
                        this.setSize([this.size[0], newSize[1]]);
                        this.setDirtyCanvas(true, true);
                    });
                };

                // ОБНОВЛЕНИЕ ИМЕН ВЫХОДОВ
                this.updateOutputInfo = () => updateNodeOutputInfo(this);

                // ---- ОБРАБОТЧИК ИЗМЕНЕНИЙ ВИДЖЕТОВ ----
                const handleWidgetChange = (widgetName, value) => {
                    // Категории действий
                    const needsPreview = ['video_path', 'timestamp', 'duration_sec', 'target_fps',
                                          'longer_size', 'megapixels', 'size_multiple'];
                    const needsCropBoxUpdate = ['crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'enable_crop'];
                    const needsOutputUpdate = ['duration_sec', 'target_fps', 'longer_size', 'megapixels',
                                               'size_multiple', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
                                               'enable_crop', 'enable_resize'];

                    try {
                        // Обновление рамки
                        if (needsCropBoxUpdate.includes(widgetName) && typeof this.updateCropBoxVisuals === 'function') {
                            this.updateCropBoxVisuals();
                        }

                        // enable_crop
                        if (widgetName === 'enable_crop' && typeof this.toggleCropWidgets === 'function') {
                            this.toggleCropWidgets(!!value);
                        }

                        // enable_resize
                        if (widgetName === 'enable_resize' && typeof this.toggleResizeWidgets === 'function') {
                            this.toggleResizeWidgets(!!value);
                        }

                        // Обновление выходов
                        if (needsOutputUpdate.includes(widgetName) && typeof this.updateOutputInfo === 'function') {
                            this.updateOutputInfo();
                        }

                        // Запрос превью
                        if (needsPreview.includes(widgetName) && typeof this.requestLivePreview === 'function') {
                            this.requestLivePreview();
                        }
                    } catch (e) {
                        console.warn(`[VideoFragment] Error in handleWidgetChange for ${widgetName}:`, e);
                    }
                };

                // ---- ПЕРЕОПРЕДЕЛЯЕМ КОЛБЭКИ ВСЕХ ВИДЖЕТОВ ----
                const allWidgetNames = ['video_path', 'timestamp', 'duration_sec', 'target_fps',
                                        'enable_resize', 'longer_size', 'megapixels', 'size_multiple',
                                        'enable_crop', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2'];
                allWidgetNames.forEach(widgetName => {
                    const widget = this.widgets.find(w => w.name === widgetName);
                    if (widget) {
                        const origCallback = widget.callback;
                        widget.callback = function(value) {
                            // Вызываем оригинальный колбэк с контекстом ВИДЖЕТА
                            if (origCallback) {
                                origCallback.call(widget, value);
                            }
                            // Теперь вызываем обработчик с контекстом НОДЫ
                            handleWidgetChange(widgetName, value);
                        }.bind(this);
                    }
                });

                // ---- ПЕРВИЧНОЕ ОБНОВЛЕНИЕ ----
                setTimeout(() => {
                    if (this.updateOutputInfo) this.updateOutputInfo();
                    if (this.updateCropBoxVisuals) this.updateCropBoxVisuals();
                    const enableCropWidget = this.widgets.find(w => w.name === 'enable_crop');
                    if (enableCropWidget && !enableCropWidget.value) {
                        this.toggleCropWidgets(false);
                    }
                    const enableResizeWidget = this.widgets.find(w => w.name === 'enable_resize');
                    if (enableResizeWidget && !enableResizeWidget.value) {
                        this.toggleResizeWidgets(false);
                    }
                    // Загружаем превью, если выбран файл
                    if (pathWidget && pathWidget.value) {
                        this.requestLivePreview();
                    }
                }, 100);

                
                // ИНИЦИАЛИЗАЦИЯ ВЗАИМОДЕЙСТВИЯ С РАМКОЙ КРОПА
                initCropInteractions(this);

                // ---- ЗАВЕРШЕНИЕ ----
                this._isPreviewPlaying = false;
            };

            nodeType.prototype.requestLivePreview = async function() {
                const videoWidget = this.widgets.find(w => w.name === "video_path");
                const timestampWidget = this.widgets.find(w => w.name === "timestamp");
                const durationWidget = this.widgets.find(w => w.name === "duration_sec");
                const fpsWidget = this.widgets.find(w => w.name === "target_fps");
                const longerWidget = this.widgets.find(w => w.name === "longer_size");
                const megapixelsWidget = this.widgets.find(w => w.name === "megapixels");
                const multipleWidget = this.widgets.find(w => w.name === "size_multiple");

                if (!videoWidget || !timestampWidget || !videoWidget.value) return;

                // 1. ОТМЕНЯЕМ ПРЕДЫДУЩИЙ ЗАПРОС, ЕСЛИ ОН ЕЩЕ ВЫПОЛНЯЕТСЯ
                if (this._previewAbortController) {
                    this._previewAbortController.abort();
                }

                // ОСТАНОВКА СТАРОГО АУДИО ПЕРЕД НОВЫМ ЗАПРОСОМ
                /*if (this._audioEl) {
                    this._audioEl.pause();
                    if (this._audioEl.src) {
                        URL.revokeObjectURL(this._audioEl.src);
                        this._audioEl.removeAttribute('src');
                        this._audioEl.load(); // сбрасываем состояние элемента
                    }
                }*/
                
                // 2. СОЗДАЕМ НОВЫЙ КОНТРОЛЛЕР ДЛЯ ТЕКУЩЕГО ЗАПРОСА
                this._previewAbortController = new AbortController();

                const timestamp = timestampWidget.value.trim();
                let start_time_sec = this._timestampToSeconds(timestamp);

                const duration = parseFloat(durationWidget?.value) || 0.0;
                const targetFps = parseFloat(fpsWidget?.value) || 16.0;
                const singleFrame = !this._isPreviewPlaying;

                if (this._total_duration !== undefined && this._total_duration > 0) {
                    // Разрешаем навигацию, но ограничиваем начало пределами видео
                    if (start_time_sec > this._total_duration) {
                        start_time_sec = this._total_duration;
                        timestampWidget.value = this._secondsToTimestamp(start_time_sec);
                    }
                }

                const configuredPreviewFps = this.properties.preview_fps || 24;
                const finalPreviewFps = configuredPreviewFps > 0 ? configuredPreviewFps : targetFps;

                const payload = {
                    video_path: videoWidget.value,
                    start_time_sec: start_time_sec,
                    duration_sec: duration,
                    single_frame: singleFrame,

                    max_preview_frames: this.properties?.preview_max_frames || 1000,
                    preview_longer_size: this.properties.preview_longer_size || 400,
                    preview_megapixels: this.properties.preview_megapixels || 0.0,
                    preview_jpeg_quality: this.properties.preview_jpeg_quality || 70,
                    preview_fps: finalPreviewFps,

                    node_id: String(this.id),
                    request_audio: !this._isMuted && this._isPreviewPlaying,
                };
                                
                try {
                    const resp = await fetch("/video_fragment/live_preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                        signal: this._previewAbortController.signal
                    });

                    const data = await resp.json();                

                    if (data.status === "success" && data.frames && data.frames.length > 0) {
                        if (this._animationFrameId) cancelAnimationFrame(this._animationFrameId);

                        this._orig_width = data.orig_width !== undefined ? data.orig_width : "?";
                        this._orig_height = data.orig_height !== undefined ? data.orig_height : "?";
                        this._actual_source_fps = data.source_fps;
                        this._total_duration = data.total_duration;
                        this._sample_rate = data.sample_rate !== undefined ? data.sample_rate : "?";

                        // ОБРАБОТКА ПОЛУЧЕННОГО АУДИО
                        if (data.audio_base64 && this._audioEl) {
                            try {
                                // ОЧИСТКА ПРЕДЫДУЩЕГО BLOB
                                if (this._audioEl.src) {
                                    URL.revokeObjectURL(this._audioEl.src);
                                }

                                const binaryString = atob(data.audio_base64);
                                const bytes = new Uint8Array(binaryString.length);
                                for (let i = 0; i < binaryString.length; i++) {
                                    bytes[i] = binaryString.charCodeAt(i);
                                }
                                const audioBlob = new Blob([bytes], { type: 'audio/wav' });
                                
                                this._audioEl.src = URL.createObjectURL(audioBlob);
                                this._audioEl.currentTime = 0;
                                this._audioEl.load();
                                
                                if (this._isPreviewPlaying && !this._isMuted) {
                                    this._audioEl.play().catch(e => console.warn("Audio play blocked:", e));
                                }
                            } catch (e) {
                                console.error("[Load Video Fragment] Ошибка при обработке аудио:", e);
                            }
                        }

                        // ЕСЛИ ВИДЕО БЕЗ ЗВУКА - ОЧИЩАЕМ СТАРОЕ АУДИО
                        if (!data.audio_base64 && this._audioEl) {
                            this._audioEl.pause();
                            if (this._audioEl.src) {
                                URL.revokeObjectURL(this._audioEl.src);
                                this._audioEl.removeAttribute('src');
                                this._audioEl.load();
                            }
                        }

                        const images = [];
                        let loadedCount = 0;
                        const onImageLoad = () => {
                            loadedCount++;
                            if (loadedCount === images.length) {
                                this._previewImages = images;
                                this._currentFrameIndex = 0;
                                this._lastFrameTime = performance.now();

                                // СБРОС ПОЛОСКИ ПРИ ЗАГРУЗКЕ НОВЫХ КАДРОВ
                                if (this._preview && this._preview.progressBar) {
                                    this._preview.progressBar.style.width = '0%';
                                }

                                const effectiveDuration = data.effective_duration || duration;
                                const realTimeFps = (effectiveDuration > 0 && !singleFrame) ? (data.count / effectiveDuration) : targetFps;
                                const displayFps = Math.min(realTimeFps, 30);
                                this._frameInterval = 1000 / displayFps;

                                if (this._preview && this._preview.img) {
                                    this._preview.img.src = images[0].src;

                                    // ПРОВЕРКА СМЕНЫ ВИДЕО
                                    const videoPathChanged = this._currentVideoPath !== videoWidget.value;
                                    if (videoPathChanged) {
                                        this._currentVideoPath = videoWidget.value;
                                        // Пересчитываем размер ноды
                                        requestAnimationFrame(() => {
                                            const newSize = this.computeSize();
                                            this.setSize([this.size[0], newSize[1]]);
                                            this.setDirtyCanvas(true, true);
                                        });
                                    }

                                }
                                if (this._isPreviewPlaying) {
                                    this._lastFrameTime = performance.now();
                                    this._animatePreview();
                                }
                                this.updateOutputInfo();
                            }
                        };
                        for (const dataUrl of data.frames) {
                            const img = new Image();
                            img.onload = onImageLoad;
                            img.onerror = () => {
                                console.error("Failed to load frame:", dataUrl);
                                onImageLoad(); // Все равно продолжаем
                            };
                            img.src = dataUrl;
                            images.push(img);
                        }

                        if (this.timelineWidget?.update) {
                            const totalVideoDuration = data.total_duration || 1;
                            const totalVideoFrames = data.total_frames || 1;
                            const sourceFps = data.source_fps || 30; 
                            this.timelineWidget.update(start_time_sec, duration, totalVideoDuration, totalVideoFrames, sourceFps);
                        }
                    } else if (data.status === "skip" && data.total_duration) {
                        // Сохраняем длительность
                        this._total_duration = data.total_duration;

                        const maxStart = Math.max(0, this._total_duration - duration);
                        // Получаем текущее время из виджета (может быть уже изменено)
                        let currentStart = 0.0;
                        const currentTs = timestampWidget.value.trim();
                        if (currentTs && currentTs !== "0:00:00") {
                            let ms = 0, mainPart = currentTs;
                            if (currentTs.includes('.')) {
                                const [main, msPart] = currentTs.split('.', 2);
                                ms = parseFloat('0.' + msPart);
                                mainPart = main;
                            }
                            const parts = mainPart.split(':');
                            if (parts.length === 1) currentStart = parseFloat(parts[0]) + ms;
                            else if (parts.length === 2) currentStart = parseInt(parts[0]) * 60 + parseFloat(parts[1]) + ms;
                            else if (parts.length === 3) currentStart = parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2]) + ms;
                        }
                        if (currentStart > maxStart && !this._isAdjusting) {
                            const newTs = this._secondsToTimestamp(maxStart);
                            timestampWidget.value = newTs;
                            this._isAdjusting = true;
                            await this.requestLivePreview();
                            this._isAdjusting = false;
                        }
                    }
                } catch (e) {
                    // ИГНОРИРУЕМ ОШИБКУ ОТМЕНЫ, ЭТО НОРМАЛЬНОЕ ПОВЕДЕНИЕ
                    if (e.name === 'AbortError') {
                        console.log("[Load Video Fragment] Request cancelled.");
                        return; 
                    }
                    console.error("[Load Video Fragment] Preview failed:", e);
                }
            };

            nodeType.prototype._animatePreview = function() {
                if (!this._isPreviewPlaying) return;
                if (!this._previewImages || this._previewImages.length === 0) return;
                if (!this._preview || !this._preview.img) return;
                
                const now = performance.now();
                const elapsed = now - this._lastFrameTime;

                if (now - this._lastFrameTime >= this._frameInterval) {
                    const nextIndex = this._currentFrameIndex + 1;
                    
                    // ДЕТЕКЦИЯ ЗАЦИКЛИВАНИЯ: следующий кадр = 0
                    const isLooping = nextIndex >= this._previewImages.length;
                    
                    this._currentFrameIndex = nextIndex % this._previewImages.length;
                    
                    if (this._preview && this._preview.img) {
                        this._preview.img.src = this._previewImages[this._currentFrameIndex].src;

                        // ОБНОВЛЕНИЕ ПОЛОСКИ ПРОГРЕССА
                        if (this._preview.progressBar) {
                            const progress = ((this._currentFrameIndex + 1) / this._previewImages.length) * 100;
                            this._preview.progressBar.style.width = `${progress}%`;
                        }
                    }
                    
                    // СИНХРОНИЗАЦИЯ ТОЛЬКО НА КАДРЕ 0 (при зацикливании)
                    if (isLooping && this._audioEl && this._audioEl.src && !this._isMuted) {
                        console.log(`[Load Video Fragment] Video loop`);
                        
                        // Сбрасываем в начало
                        try {
                            this._audioEl.currentTime = 0;
                        } catch (e) { }
                        
                        // 🎯 ЯВНО ЗАПУСКАЕМ ВОСПРОИЗВЕДЕНИЕ, если аудио остановилось
                        if (this._audioEl.paused || this._audioEl.ended) {
                            this._audioEl.play().catch(e => console.warn("Audio loop restart blocked:", e));
                        }
                    }            

                    // Если разница слишком большая (вкладка была свёрнута), сбрасываем таймер
                    if (elapsed > this._frameInterval * 3) {
                        // Вкладка была неактивна — сбрасываем, чтобы не было ускоренной перемотки
                        this._lastFrameTime = now;
                    } else {
                        // Нормальный режим — компенсируем дрейф
                        this._lastFrameTime += this._frameInterval;
                    }

                }
                this._animationFrameId = requestAnimationFrame(() => this._animatePreview());
            };
        }
    }
});

// ========================================================================
// МОДУЛЬ УПРАВЛЕНИЯ КРОПОМ 
// ========================================================================

function getCropCanvasScale() {
    if (typeof app !== 'undefined' && app.canvas && app.canvas.ds) {
        return app.canvas.ds.scale || 1;
    }
    return 1;
}

// --- Обновление виджетов (значения 0.0-1.0) --- 
function updateCropWidgets(hostNode, leftPct, topPct, widthPct, heightPct) {
    hostNode._suppressCropCallback = true;
    
    const wX1 = hostNode.widgets.find(w => w.name === 'crop_x1');
    const wY1 = hostNode.widgets.find(w => w.name === 'crop_y1');
    const wX2 = hostNode.widgets.find(w => w.name === 'crop_x2');
    const wY2 = hostNode.widgets.find(w => w.name === 'crop_y2');

    // Жёсткая нормализация: гарантируем [0, 1]
    leftPct = Math.max(0, Math.min(1.0, leftPct));
    topPct = Math.max(0, Math.min(1.0, topPct));
    widthPct = Math.max(0, Math.min(1.0 - leftPct, widthPct));
    heightPct = Math.max(0, Math.min(1.0 - topPct, heightPct));

    if (wX1) wX1.value = Math.round(leftPct * 100000) / 100000;
    if (wY1) wY1.value = Math.round(topPct * 100000) / 100000;
    if (wX2) wX2.value = Math.round((leftPct + widthPct) * 100000) / 100000;
    if (wY2) wY2.value = Math.round((topPct + heightPct) * 100000) / 100000;
    
    hostNode._suppressCropCallback = false;
}

// ---  Чтение текущего состояния рамки (в процентах 0.0-1.0) --- 
function getCurrentCropState(hostNode) {
    const container = hostNode._preview.container;
    const cropBox = hostNode._preview.cropBox;

    // Читаем проценты из style (parseFloat("33.33%") → 33.33) и делим на 100
    let left = parseFloat(cropBox.style.left) / 100;
    let top = parseFloat(cropBox.style.top) / 100;
    let width = parseFloat(cropBox.style.width) / 100;
    let height = parseFloat(cropBox.style.height) / 100;

    // Если стили ещё не заданы — берём из виджетов
    if (isNaN(left) || isNaN(width)) {
        const wX1 = hostNode.widgets.find(w => w.name === 'crop_x1')?.value || 0;
        const wY1 = hostNode.widgets.find(w => w.name === 'crop_y1')?.value || 0;
        const wX2 = hostNode.widgets.find(w => w.name === 'crop_x2')?.value || 1;
        const wY2 = hostNode.widgets.find(w => w.name === 'crop_y2')?.value || 1;

        left = Math.min(wX1, wX2);
        top = Math.min(wY1, wY2);
        width = Math.abs(wX2 - wX1);
        height = Math.abs(wY2 - wY1);
    }

    const MIN_SIZE = 0.02; //2% от контейнера
    width = Math.max(MIN_SIZE, Math.min(width, 1.0));
    height = Math.max(MIN_SIZE, Math.min(height, 1.0));
    left = Math.max(0, Math.min(left, 1.0 - width));
    top = Math.max(0, Math.min(top, 1.0 - height));

    return { left, top, width, height };
}

// ---  Завершение действия (drag или resize) --- 
function finalizeCropAction(state, hostNode) {
    const target = state.activeTarget;
    if (!target) return;

    target.releasePointerCapture(state.pointerId);
    target.removeEventListener("pointermove", state.boundMove);
    target.removeEventListener("pointerup", state.boundUp);

    state.isDragging = false;
    state.isResizing = false;
    state.resizeDirection = null;
    state.activeTarget = null;
    state.pointerId = null;
    state.boundMove = null;
    state.boundUp = null;

    hostNode.updateOutputInfo();
}


// --- ПЕРЕТАСКИВАНИЕ РАМКИ ---
function onCropBoxPointerDown(e, hostNode) {
    const cropBox = hostNode._preview.cropBox;
    if (e.target !== cropBox) return;

    const state = hostNode._cropState;
    const currentState = getCurrentCropState(hostNode);
    
    state.isDragging = true;
    state.activeTarget = cropBox;
    state.pointerId = e.pointerId;
    state.pointerStartX = e.clientX;
    state.pointerStartY = e.clientY;
    state.initialLeft = currentState.left;
    state.initialTop = currentState.top;
    state.initialWidth = currentState.width;
    state.initialHeight = currentState.height;

    cropBox.setPointerCapture(e.pointerId);
    state.boundMove = (ev) => onCropBoxPointerMove(ev, hostNode);
    state.boundUp = (ev) => finalizeCropAction(state, hostNode);

    cropBox.addEventListener("pointermove", state.boundMove);
    cropBox.addEventListener("pointerup", state.boundUp);
    
    e.preventDefault();
}

function onCropBoxPointerMove(e, hostNode) {
    const state = hostNode._cropState;
    const container = hostNode._preview.container;
    const cropBox = hostNode._preview.cropBox;
    if (!cropBox || !container) return;

    const scale = getCropCanvasScale();
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    if (containerWidth === 0 || containerHeight === 0) return;

    // 🎯 Конвертируем пиксельную дельту в проценты
    const deltaX = (e.clientX - state.pointerStartX) / scale / containerWidth;
    const deltaY = (e.clientY - state.pointerStartY) / scale / containerHeight;
    
    const maxLeft = 1.0 - state.initialWidth;
    const maxTop = 1.0 - state.initialHeight;
    
    const newLeft = Math.max(0, Math.min(maxLeft, state.initialLeft + deltaX));
    const newTop = Math.max(0, Math.min(maxTop, state.initialTop + deltaY));

    // 🎯 Сразу в процентах — CSS сам отрисует
    cropBox.style.left = `${newLeft * 100}%`;
    cropBox.style.top = `${newTop * 100}%`;

    updateCropWidgets(hostNode, newLeft, newTop, state.initialWidth, state.initialHeight);
}


// --- ИЗМЕНЕНИЕ РАЗМЕРА (УГЛОВЫЕ РУЧКИ) ---
function onHandlePointerDown(e, hostNode, direction) {
    const state = hostNode._cropState;
    const currentState = getCurrentCropState(hostNode);
    
    state.isResizing = true;
    state.resizeDirection = direction;
    state.activeTarget = e.currentTarget;
    state.pointerId = e.pointerId;
    state.pointerStartX = e.clientX;
    state.pointerStartY = e.clientY;
    state.initialLeft = currentState.left;
    state.initialTop = currentState.top;
    state.initialWidth = currentState.width;
    state.initialHeight = currentState.height;

    e.currentTarget.setPointerCapture(e.pointerId);
    state.boundMove = (ev) => onHandlePointerMove(ev, hostNode);
    state.boundUp = (ev) => finalizeCropAction(state, hostNode);

    e.currentTarget.addEventListener("pointermove", state.boundMove);
    e.currentTarget.addEventListener("pointerup", state.boundUp);
    
    e.preventDefault();
    e.stopPropagation();
}

function onHandlePointerMove(e, hostNode) {
    const state = hostNode._cropState;
    const container = hostNode._preview.container;
    const cropBox = hostNode._preview.cropBox;
    if (!cropBox || !container) return;

    const scale = getCropCanvasScale();
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    if (containerWidth === 0 || containerHeight === 0) return;

    // 🎯 Конвертируем пиксельную дельту в проценты
    const deltaX = (e.clientX - state.pointerStartX) / scale / containerWidth;
    const deltaY = (e.clientY - state.pointerStartY) / scale / containerHeight;

    const MIN_SIZE = 0.02; //2% от контейнера
    let newLeft = state.initialLeft;
    let newTop = state.initialTop;
    let newWidth = state.initialWidth;
    let newHeight = state.initialHeight;

    const dir = state.resizeDirection;

    // Горизонтальное изменение
    if (dir.includes('left')) {
        newLeft = state.initialLeft + deltaX;
        newWidth = state.initialWidth - deltaX;
        if (newWidth < MIN_SIZE) {
            newWidth = MIN_SIZE;
            newLeft = state.initialLeft + state.initialWidth - MIN_SIZE;
        }
        if (newLeft < 0) {
            newWidth += newLeft;
            newLeft = 0;
        }
    } else {
        newWidth = state.initialWidth + deltaX;
        if (newWidth < MIN_SIZE) newWidth = MIN_SIZE;
        if (state.initialLeft + newWidth > 1.0) {
            newWidth = 1.0 - state.initialLeft;
        }
    }

    // Вертикальное изменение
    if (dir.includes('top')) {
        newTop = state.initialTop + deltaY;
        newHeight = state.initialHeight - deltaY;
        if (newHeight < MIN_SIZE) {
            newHeight = MIN_SIZE;
            newTop = state.initialTop + state.initialHeight - MIN_SIZE;
        }
        if (newTop < 0) {
            newHeight += newTop;
            newTop = 0;
        }
    } else {
        newHeight = state.initialHeight + deltaY;
        if (newHeight < MIN_SIZE) newHeight = MIN_SIZE;
        if (state.initialTop + newHeight > 1.0) {
            newHeight = 1.0 - state.initialTop;
        }
    }

    // 🎯 Сразу в процентах
    cropBox.style.left = `${newLeft * 100}%`;
    cropBox.style.top = `${newTop * 100}%`;
    cropBox.style.width = `${newWidth * 100}%`;
    cropBox.style.height = `${newHeight * 100}%`;

    updateCropWidgets(hostNode, newLeft, newTop, newWidth, newHeight);
}


// --- Главная функция инициализации --- 
function initCropInteractions(hostNode) {
    if (!hostNode._preview || !hostNode._preview.cropBox) return;
    
    hostNode._cropState = {
        isDragging: false,
        isResizing: false,
        resizeDirection: null,
        pointerStartX: 0, 
        pointerStartY: 0,
        initialLeft: 0, 
        initialTop: 0,
        initialWidth: 0, 
        initialHeight: 0,
        activeTarget: null,
        pointerId: null,
        boundMove: null, 
        boundUp: null
    };

    const cropBox = hostNode._preview.cropBox;
    cropBox.addEventListener("pointerdown", (e) => onCropBoxPointerDown(e, hostNode));

    ['top-left', 'top-right', 'bottom-left', 'bottom-right'].forEach(dir => {
        const handle = document.createElement("div");
        handle.style.cssText = `
            position: absolute; width: 12px; height: 12px;
            background: #4a90e2; border: 1px solid #fff;
            border-radius: 50%; z-index: 10;
        `;
        
        if (dir === 'top-left') { handle.style.top = '-6px'; handle.style.left = '-6px'; handle.style.cursor = 'nwse-resize'; }
        else if (dir === 'top-right') { handle.style.top = '-6px'; handle.style.right = '-6px'; handle.style.cursor = 'nesw-resize'; }
        else if (dir === 'bottom-left') { handle.style.bottom = '-6px'; handle.style.left = '-6px'; handle.style.cursor = 'nesw-resize'; }
        else if (dir === 'bottom-right') { handle.style.bottom = '-6px'; handle.style.right = '-6px'; handle.style.cursor = 'nwse-resize'; }

        handle.addEventListener("pointerdown", (e) => onHandlePointerDown(e, hostNode, dir));
        cropBox.appendChild(handle);
    });
}

// ========================================================================
// 🎯ОДУЛЬ ОБНОВЛЕНИЯ ИНФОРМАЦИИ О ВЫХОДАХ (OUTPUT INFO)
// ========================================================================
function updateNodeOutputInfo(node) {
    try {
        // 1. Базовые запрошенные значения
        const durationWidget = node.widgets.find(w => w.name === "duration_sec");
        const fpsWidget = node.widgets.find(w => w.name === "target_fps");
        
        const requestedDuration = parseFloat(durationWidget?.value) || 0;
        const requestedFps = parseFloat(fpsWidget?.value) || 16;
        const sourceFps = node._actual_source_fps || 0;
        const totalDuration = node._total_duration || 0;

        let startSec = 0;
        if (node._timestampWidget) {
            startSec = node._timestampToSeconds(node._timestampWidget.value);
        }

        // 2. Расчет границ и кадров (как в Python execute)
        const remainingDuration = Math.max(0, totalDuration - startSec);
        const effectiveDuration = Math.min(requestedDuration, remainingDuration);

        // FPS всегда равен запрошенному, но не выше исходного
        const effectiveFps = sourceFps > 0 ? Math.min(requestedFps, sourceFps) : requestedFps;

        // Количество кадров (минимум 1)
        const numFrames = Math.max(1, Math.floor(effectiveDuration * effectiveFps));

        // 3. Расчет размеров (Crop & Resize)
        let origW = node._orig_width ?? "?";
        let origH = node._orig_height ?? "?";
        let cropW = origW;
        let cropH = origH;

        const enableCropWidget = node.widgets.find(w => w.name === 'enable_crop');
        const isCropEnabled = enableCropWidget ? !!enableCropWidget.value : false;

        const enableResizeWidget = node.widgets.find(w => w.name === 'enable_resize');
        const isResizeEnabled = enableResizeWidget ? !!enableResizeWidget.value : false;

        if (origW !== "?" && origH !== "?" && isCropEnabled) {
            const getVal = (name, fallback) => {
                const w = node.widgets.find(wid => wid.name === name);
                const val = w ? parseFloat(w.value) : fallback;
                return isNaN(val) ? fallback : val;
            };
            const x1 = getVal('crop_x1', 0.0);
            const y1 = getVal('crop_y1', 0.0);
            const x2 = getVal('crop_x2', 1.0);
            const y2 = getVal('crop_y2', 1.0);

            cropW = Math.floor(Math.abs(x2 - x1) * origW);
            cropH = Math.floor(Math.abs(y2 - y1) * origH);
        }

        let finalW = cropW;
        let finalH = cropH;

        if (cropW !== "?" && cropH !== "?" && isResizeEnabled) {
            const longerSize = parseInt(node.widgets.find(w => w.name === "longer_size")?.value) || 0;
            const megapixels = parseFloat(node.widgets.find(w => w.name === "megapixels")?.value) || 0.0;
            const sizeMultiple = parseInt(node.widgets.find(w => w.name === "size_multiple")?.value) || 2;

            if (longerSize > 0) {
                const scale = longerSize / Math.max(cropW, cropH);
                finalW = Math.floor(cropW * scale);
                finalH = Math.floor(cropH * scale);
            } else if (megapixels > 0) {
                const scale = Math.sqrt((megapixels * 1_000_000) / (cropW * cropH));
                finalW = Math.floor(cropW * scale);
                finalH = Math.floor(cropH * scale);
            }

            if (sizeMultiple > 1) {
                finalW = Math.floor(finalW / sizeMultiple) * sizeMultiple;
                finalH = Math.floor(finalH / sizeMultiple) * sizeMultiple;
            }
        }

        // 4. Обновление лейблов выходов
        if (node.outputs && node.outputs.length >= 8) {
            node.outputs[0].label = "frames";
            node.outputs[1].label = `count: ${numFrames}`;
            node.outputs[2].label = `duration: ${requestedDuration}s -> ${effectiveDuration.toFixed(2)}s`;
            node.outputs[3].label = `fps: ${sourceFps > 0 ? sourceFps.toFixed(3) : "?"} -> ${effectiveFps.toFixed(2)}`;

            if (origW == "?" && origH == "?") {
                node.outputs[4].label = `width`;
                node.outputs[5].label = `height`;
            } else {
                if (isCropEnabled) {
                    if (isResizeEnabled) {
                        node.outputs[4].label = `width: ${origW} -> crop ${cropW} -> resize ${finalW}`;
                        node.outputs[5].label = `height: ${origH} -> crop ${cropH} -> resize ${finalH}`;
                    } else {
                        node.outputs[4].label = `width: ${origW} -> crop ${cropW}`;
                        node.outputs[5].label = `height: ${origH} -> crop ${cropH}`;
                    }
                } else {
                    if (isResizeEnabled) {
                        node.outputs[4].label = `width: ${origW} -> resize ${finalW}`;
                        node.outputs[5].label = `height: ${origH} -> resize ${finalH}`;
                    } else {
                        node.outputs[4].label = `width: ${origW}`;
                        node.outputs[5].label = `height: ${origH}`;
                    }
                }
            }

            node.outputs[7].label = node._sample_rate != null ? `sample_rate: ${node._sample_rate}` : `sample_rate`;
        }

        // Сообщаем ComfyUI что холст нужно перерисовать, чтобы обновить текст на выходах
        //node.setDirtyCanvas(true, true);
        //if (node.graph) node.graph.setDirtyCanvas(true, true);

    } catch (e) {
        console.warn("[VideoFragment] updateOutputInfo error:", e);
    }
}

// === PREVIEW WIDGET ===
function createPreviewWidget(hostNode) {
    const container = document.createElement("div");
    container.style.cssText = `
        position: relative;
        width: 100%;
        background: #000;
        border-radius: 4px;
        margin: 8px 0;
        overflow: hidden;
        user-select: none;
    `;

    const img = document.createElement("img");
    img.style.cssText = `
        display: block;
        width: 100%;
        height: auto; /* Позволяет изображению масштабироваться по Y */
        object-fit: contain;
        pointer-events: none;
    `;
    container.appendChild(img);

    const cropBox = document.createElement("div");
    cropBox.style.cssText = `
        position: absolute;
        border: 2px dashed #4a90e2;
        background: rgba(74, 144, 226, 0.2);
        cursor: move;
        display: none;
        box-sizing: border-box;
    `;    
    container.appendChild(cropBox);

    // ПОЛОСКА ПРОГРЕССА ВОСПРОИЗВЕДЕНИЯ
    const progressBar = document.createElement("div");
    progressBar.style.cssText = `
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        width: 0%;
        background: linear-gradient(90deg, #4a90e2 0%, #00ff00 100%);
        opacity: 0.8;
        transition: width 0.05s linear;
        z-index: 5;
        pointer-events: none;
    `;
    container.appendChild(progressBar);

    // КНОПКА PLAY (слева внизу, поверх видео)
    const playBtn = document.createElement("button");
    playBtn.textContent = "▶";
    playBtn.style.cssText = `
        position: absolute;
        bottom: 8px;
        left: 8px;
        width: 32px;
        height: 32px;
        background: rgba(0, 0, 0, 0.6);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        cursor: pointer;
        font-size: 14px;
        line-height: 1;
        padding: 0;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        backdrop-filter: blur(4px);
    `;
    playBtn.addEventListener("mouseenter", () => {
        playBtn.style.background = "rgba(74, 144, 226, 0.8)";
        playBtn.style.borderColor = "#4a90e2";
    });
    playBtn.addEventListener("mouseleave", () => {
        playBtn.style.background = "rgba(0, 0, 0, 0.6)";
        playBtn.style.borderColor = "rgba(255, 255, 255, 0.3)";
    });
    playBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        hostNode._isPreviewPlaying = !hostNode._isPreviewPlaying;
        
        if (hostNode._isPreviewPlaying) {
            playBtn.textContent = "⏸";
            if (hostNode.requestLivePreview) {
                await hostNode.requestLivePreview();
            }
        } else {
            playBtn.textContent = "▶";
            if (hostNode._animationFrameId) {
                cancelAnimationFrame(hostNode._animationFrameId);
                hostNode._animationFrameId = null;
            }
            if (hostNode._audioEl) {
                hostNode._audioEl.pause();
            }
            if (hostNode._previewImages && hostNode._previewImages.length > 0) {
                img.src = hostNode._previewImages[0].src;
                if (progressBar) progressBar.style.width = '0%';
            }
        }
    });
    container.appendChild(playBtn);
    hostNode._playPauseBtn = playBtn;

    // КНОПКА MUTE (справа внизу, поверх видео)
    const muteBtn = document.createElement("button");
    hostNode._isMuted = true;
    muteBtn.textContent = "🔇";
    muteBtn.style.cssText = `
        position: absolute;
        bottom: 8px;
        right: 8px;
        width: 32px;
        height: 32px;
        background: rgba(0, 0, 0, 0.6);
        color: #cccccc;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        cursor: pointer;
        font-size: 14px;
        line-height: 1;
        padding: 0;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        backdrop-filter: blur(4px);
    `;
    muteBtn.addEventListener("mouseenter", () => {
        muteBtn.style.background = "rgba(74, 144, 226, 0.8)";
        muteBtn.style.borderColor = "#4a90e2";
    });
    muteBtn.addEventListener("mouseleave", () => {
        muteBtn.style.background = "rgba(0, 0, 0, 0.6)";
        muteBtn.style.borderColor = "rgba(255, 255, 255, 0.3)";
    });
    muteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        hostNode._isMuted = !hostNode._isMuted;
        muteBtn.textContent = hostNode._isMuted ? "🔇" : "🔊";
        muteBtn.style.color = hostNode._isMuted ? "#cccccc" : "#4a90e2";
        
        if (hostNode._audioEl) {
            hostNode._audioEl.muted = hostNode._isMuted;
        }
        
        if (!hostNode._isMuted) {
            hostNode.requestLivePreview();
        }
    });
    container.appendChild(muteBtn);
    hostNode._muteBtn = muteBtn;

    // СКРЫТЫЙ АУДИО ЭЛЕМЕНТ
    const audioEl = document.createElement("audio");
    audioEl.style.display = "none";
    audioEl.muted = true;
    audioEl.loop = false;
    hostNode._audioEl = audioEl;
    document.body.appendChild(audioEl);

    const widget = hostNode.addDOMWidget("crop_preview", "vf_crop_preview", container, {
        serialize: false,
        hideOnZoom: false,
    });

    // РЕСАЙЗ: используем clientHeight (реальный размер на экране)
    widget.computeSize = function(width) {
        let imgHeight = img.clientHeight;    
        if (imgHeight > 0) {
            widget._cachedHeight = imgHeight + 16;
            return [width, widget._cachedHeight];
        }
        return [width, 200];
    };

    // ФУНКЦИЯ ОБНОВЛЕНИЯ РАМКИ
    hostNode.updateCropBoxVisuals = () => {
        const enableWidget = hostNode.widgets.find(w => w.name === 'enable_crop');
        const isEnabled = enableWidget ? !!enableWidget.value : false;

        if (!isEnabled || !img.clientWidth) {
            cropBox.style.display = "none";
            return;
        }

        const getVal = (name, fallback) => {
            const w = hostNode.widgets.find(wid => wid.name === name);
            let val = w ? parseFloat(w.value) : fallback;
            if (isNaN(val)) val = fallback;
            return val;
        };

        const x1 = getVal('crop_x1', 0.0);
        const y1 = getVal('crop_y1', 0.0);
        const x2 = getVal('crop_x2', 1.0);
        const y2 = getVal('crop_y2', 1.0);

        // Нормализация: гарантируем, что значения в пределах [0, 1]
        const left = Math.max(0, Math.min(x1, x2));
        const top = Math.max(0, Math.min(y1, y2));
        const right = Math.min(1.0, Math.max(x1, x2));
        const bottom = Math.min(1.0, Math.max(y1, y2));

        const width = Math.max(0, right - left);
        const height = Math.max(0, bottom - top);

        if (width <= 0 || height <= 0) {
            cropBox.style.display = "none";
            return;
        }

        cropBox.style.display = "block";
        
        cropBox.style.left = `${left * 100}%`;
        cropBox.style.top = `${top * 100}%`;
        cropBox.style.width = `${width * 100}%`;
        cropBox.style.height = `${height * 100}%`;
    };

    return { widget, container, img, cropBox, progressBar };
}

// === TIMELINE WIDGET ===
function createTimelineWidget(hostNode) {
    const element = document.createElement("div");
    element.className = "vf-timeline";
    element.style.cssText = `
        width: 100%;
        height: 28px;
        background: #1a1a1a;
        border-radius: 4px;
        margin: 8px 0;
        position: relative;
        cursor: pointer;
        overflow: hidden;
    `;

    // Track (фон)
    const trackEl = document.createElement("div");
    trackEl.style.cssText = `
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background: #2a2a2a;
    `;
    element.appendChild(trackEl);

    // PRE-FILL (светло-зелёная область до начала фрагмента)
    const preFillEl = document.createElement("div");
    preFillEl.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: #90EE90;
        opacity: 0.4;
        z-index: 1;
    `;
    element.appendChild(preFillEl);

    // Fill (активная область фрагмента)
    const fillEl = document.createElement("div");
    fillEl.style.cssText = `
        position: absolute;
        top: 0;
        height: 100%;
        background: #4a90e2;
        opacity: 0.6;
        z-index: 1;
    `;
    element.appendChild(fillEl);

    // Start marker (линия начала)
    const startMarkerEl = document.createElement("div");
    startMarkerEl.style.cssText = `
        position: absolute;
        top: 0;
        width: 2px;
        height: 100%;
        background: #00ff00;
        z-index: 2;
    `;
    element.appendChild(startMarkerEl);

    // Label (полная информация)
    const labelEl = document.createElement("div");
    labelEl.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #cccccc;
        font-size: 14px;
        font-family: 'Consolas', 'Monaco', monospace;
        white-space: nowrap;
        z-index: 3;
        pointer-events: none;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.9);
    `;
    labelEl.textContent = "0:00:00 / 0:00:00 | Frame 1 / 1";
    element.appendChild(labelEl);

    // Создаем DOM-виджет
    const timelineWidget = hostNode.addDOMWidget("timeline", "vf_timeline", element, {
        serialize: false,
        hideOnZoom: false,
    });
    
    timelineWidget.computeSize = function(width) {
        return [width, 36];
    };

    // Сохраняем общую длительность видео
    timelineWidget.totalVideoDuration = 0;
    timelineWidget.totalVideoFrames = 0;

    // Обработка кликов по таймлайну
    element.addEventListener("mousedown", (e) => {
        const rect = element.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = x / rect.width;
        const newTime = pct * (timelineWidget.totalVideoDuration || 1);
        
        if (hostNode._timestampWidget) {
            hostNode._timestampWidget.value = hostNode._secondsToTimestamp(newTime);
            hostNode._timestampWidget.callback?.(hostNode._timestampWidget.value);
        }
    });

    // Метод обновления таймлайна
    timelineWidget.update = function(startSec, duration, totalVideoDuration, totalVideoFrames, sourceFps) {
        if (!totalVideoDuration || totalVideoDuration <= 0) totalVideoDuration = 1;
        if (!totalVideoFrames || totalVideoFrames <= 0) totalVideoFrames = 1;
        if (!sourceFps || sourceFps <= 0) sourceFps = 30; 
        
        timelineWidget.totalVideoDuration = totalVideoDuration;
        timelineWidget.totalVideoFrames = totalVideoFrames;
        
        const startPct = (startSec / totalVideoDuration) * 100;
        
        const maxAvailableDuration = Math.max(0, totalVideoDuration - startSec);
        const effectiveDuration = Math.min(duration, maxAvailableDuration);
        const durationPct = Math.min(100 - startPct, Math.max(0, (effectiveDuration / totalVideoDuration) * 100));
        
        const startFrame = Math.round(startSec * sourceFps);
        
        preFillEl.style.width = `${startPct}%`;
        fillEl.style.left = `${startPct}%`;
        fillEl.style.width = `${durationPct}%`;
        startMarkerEl.style.left = `${startPct}%`;
        
        const startTimeStr = hostNode._secondsToTimestamp(startSec);
        const totalTimeStr = hostNode._secondsToTimestamp(totalVideoDuration);
        labelEl.textContent = `Start ${startTimeStr} / ${totalTimeStr} | Frame ${startFrame} / ${totalVideoFrames}`;
    };

    return timelineWidget;
}

// === SEEK CONTROLS WIDGET ===
function createSeekControlsWidget(hostNode) {
    const element = document.createElement("div");
    element.style.cssText = `
        display: grid;
        grid-template-columns: repeat(8, 1fr);
        gap: 4px;
        margin: 8px 0;
    `;

    const buttons = [
        { label: "◀◀◀ 1d", shift: "+duration" }, 
        { label: "◀◀ 10s", shift: -10 },
        { label: "◀ 1s",  shift: -1 },
        { label: "◁ 1f",  shift: "+frame" },  
        { label: "▷ 1f",  shift: "-frame" },
        { label: "▶ 1s",  shift: 1 },
        { label: "▶▶ 10s", shift: 10 },
        { label: "▶▶▶ 1d", shift: "-duration" },
    ];

    buttons.forEach(btn => {
        const button = document.createElement("button");
        button.textContent = btn.label;
        button.style.cssText = `
            background: #2a2a2a;
            color: #cccccc;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 6px 4px;
            cursor: pointer;
            font-size: 11px;
            font-family: sans-serif;
            transition: all 0.2s;
        `;
        
        button.addEventListener("mouseenter", () => {
            button.style.background = "#4a90e2";
            button.style.color = "#ffffff";
        });
        
        button.addEventListener("mouseleave", () => {
            button.style.background = "#2a2a2a";
            button.style.color = "#cccccc";
        });
        
        button.addEventListener("mousedown", () => {
            button.style.opacity = "0.7";
        });
        
        button.addEventListener("mouseup", () => {
            button.style.opacity = "1";
        });

        button.addEventListener("click", () => {
            if (!hostNode._timestampWidget) return;
            
            let shift = btn.shift;

            if (shift === "+frame") {
                const currentSourceFps = hostNode._actual_source_fps || 30;
                shift = -(1 / currentSourceFps);
            } else if (shift === "-frame") {
                const currentSourceFps = hostNode._actual_source_fps || 30;
                shift = 1 / currentSourceFps;
            }

            if (shift === "+duration") {
                const durationWidget = hostNode.widgets.find(w => w.name === "duration_sec");
                const duration = parseFloat(durationWidget?.value) || 0;
                shift = -duration;
            } else if (shift === "-duration") {
                const durationWidget = hostNode.widgets.find(w => w.name === "duration_sec");
                const duration = parseFloat(durationWidget?.value) || 0;
                shift = duration;
            }
            
            const currentSec = hostNode._timestampToSeconds(hostNode._timestampWidget.value);
            const newSec = Math.max(0, currentSec + shift);
            
            hostNode._timestampWidget.value = hostNode._secondsToTimestamp(newSec);
            hostNode._timestampWidget.callback?.(hostNode._timestampWidget.value);
        });

        element.appendChild(button);
    });

    const controlsWidget = hostNode.addDOMWidget("seek_controls", "vf_seek_controls", element, {
        serialize: false,
        hideOnZoom: false,
    });
    
    controlsWidget.computeSize = function(width) {
        return [width, 40];
    };

    return controlsWidget;
}
