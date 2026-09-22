import { app } from "../../../scripts/app.js";

const DYNAMIC_NODES = [
    {
        nodeName: "FixBatchImages",
        inputPrefixes: ["image"],
        inputTypes: { "image": "IMAGE" },
    },
    {
        nodeName: "SimpleJoinStringsNode",
        inputPrefixes: ["text"],
        inputTypes: { "text": "STRING" },
    },
    {
        nodeName: "SimpleQwenVLggufV2",
        inputPrefixes: ["image", "audio", "video"], 
        inputTypes: { 
            "image": "IMAGE",
            "audio": "AUDIO",
            "video": "*"
        },
    }
];

const LLAMA_FREE_ENDPOINT = "/simpleqwenvl/memory/free";

app.registerExtension({
    name: "DynamicInputSlots",

    // --- Хук 1: Перехват кнопки "Пылесос" ---
    async setup() {
        const originalFetchApi = app.api.fetchApi;

        app.api.fetchApi = async function(url, options) {
            const urlString = typeof url === "string" ? url : url?.url;
            
            // Проверяем, что это вызов "Пылесоса"
            if (urlString === "/free" || urlString?.endsWith("/free")) {
                // НЕ используем await!
                // Запускаем запрос в фоне, чтобы не блокировать оригинальный /free
                // и не мешать другим расширениям в цепочке перехватчиков.
                originalFetchApi.call(this, LLAMA_FREE_ENDPOINT, { 
                    method: "POST" 
                })
                .then(() => console.log("✅ [SimpleQwen] Память llama.cpp очищена"))
                .catch(e => console.warn("⚠️ [SimpleQwen] Ошибка очистки памяти:", e));
            }

            // Оригинальный вызов всегда выполнится
            return originalFetchApi.apply(this, arguments);
        };
    },

    // --- Хук 2: Динамические слоты ---
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = DYNAMIC_NODES.find(c => c.nodeName === nodeData.name);
        if (!config) return;

        const { inputPrefixes, inputTypes } = config;

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        
        nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link_info, ioSlot) {

            // Вызов оригинального обработчика (если есть)
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }

            // Обрабатываем только изменения на входах (type === 1)
            if (type !== 1) return;

            // Ранняя проверка по ioSlot
            if (!ioSlot || typeof ioSlot.name !== 'string') return;

            // Проверяем, относится ли имя слота к одному из наших префиксов
            let matched = false;
            for (const prefix of inputPrefixes) {
                if (ioSlot.name === prefix || ioSlot.name.startsWith(prefix)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) return;

            // Debounce для избежания множественных вызовов
            if (this._dynamicTimeout) {
                clearTimeout(this._dynamicTimeout);
                this._dynamicTimeout = null;
            }

            this._dynamicTimeout = setTimeout(() => {
                // Блокируем повторный вход
                if (this._processingDynamicSlots) return;
                this._processingDynamicSlots = true;

                try {

                    if (!this.inputs) return;

                    let structureChanged = false;

                    // Пробегаемся независимо по каждой группе для проверки и удаления, и добавления
                    for (const prefix of inputPrefixes) {

                        let currentIndices = [];
                        let maxNum = 1;

                        // 1. Собираем индексы и находим максимальный номер суффикса
                        for (let i = 0; i < this.inputs.length; i++) {
                            const name = this.inputs[i].name;
                            
                            if (name === prefix) {
                                // "image" считается 1
                                currentIndices.push(i);
                            } else if (name.startsWith(prefix)) {
                                const suffix = name.slice(prefix.length);
                                
                                // Проверка, что суффикс состоит только из цифр
                                let isDigits = suffix.length > 0;
                                for (let j = 0; j < suffix.length; j++) {
                                    if (suffix[j] < '0' || suffix[j] > '9') {
                                        isDigits = false;
                                        break;
                                    }
                                }
                                
                                if (isDigits) {
                                    currentIndices.push(i);
                                    const num = Number(suffix);
                                    if (num > maxNum) maxNum = num;
                                }
                            }
                        }                    

                        if (currentIndices.length === 0) continue;

                        const lastIndex = currentIndices[currentIndices.length - 1];
                        const lastInput = this.inputs[lastIndex];

                        // 1. ПРАВИЛО ДОБАВЛЕНИЯ: Если последний слот подключен — добавляем новый
                        if (lastInput && lastInput.link != null) {
                            const newName = `${prefix}${maxNum + 1}`;
                            const inputType = inputTypes[prefix] || "*";
                            
                            if (inputType === "STRING") {
                                this.addInput(newName, inputType, { multiline: true, default: "", forceInput: true });
                            } else {
                                this.addInput(newName, inputType);
                            }
                            structureChanged = true;

                            //console.log("add",newName);

                            continue; // Выходим, удалять нечего
                        }

                        if (currentIndices.length === 1) continue;

                        // 2. ПРАВИЛО УДАЛЕНИЯ: Бежим с конца до первого подключенного слота
                        const toRemove = [];
                        for (let i = currentIndices.length - 1; i >= 0; i--) {
                            const realIndex = currentIndices[i];
                            const input = this.inputs[realIndex];
                            if (input && input.link == null) {
                                toRemove.push(realIndex);
                            } else {
                                break; // Дошли до подключенного слота, стоп
                            }
                        }

                        if (toRemove.length <= 1) continue;

                        // Исключаем последний элемент из массива на удаление. 
                        toRemove.pop();         

                        // Удаляем собранные индексы (массив уже отсортирован по убыванию, это безопасно)
                        for (const idx of toRemove) {
                            this.removeInput(idx);

                            //console.log("delete",toRemove);

                            structureChanged = true;
                        }
                    }

                    if (structureChanged) {
                        this.setSize(this.computeSize());
                        this.setDirtyCanvas(true, true);
                    }

                } finally {
                    this._processingDynamicSlots = false;
                    this._dynamicTimeout = null;
                }
            }, 200); 
        };
    }
});