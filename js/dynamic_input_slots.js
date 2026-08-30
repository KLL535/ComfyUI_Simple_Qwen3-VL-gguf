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

app.registerExtension({
    name: "DynamicInputSlots",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = DYNAMIC_NODES.find(c => c.nodeName === nodeData.name);
        if (!config) return;

        const { inputPrefixes, inputTypes } = config;

        const findPrefix = (name) => {
            for (const prefix of inputPrefixes) {
                if (name === prefix || name.startsWith(prefix)) {
                    return prefix;
                }
            }
            return null;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        
        nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link_info, ioSlot) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }

            if (type !== 1) return;

            setTimeout(() => {
                if (!this.inputs) return;

                let structureChanged = false;

                // Пробегаемся независимо по каждой группе для проверки и удаления, и добавления
                for (const prefix of inputPrefixes) {
                    
                    // Собираем актуальные индексы слотов текущей группы
                    let groupIndices = [];
                    this.inputs.forEach((inp, idx) => {
                        if (findPrefix(inp.name) === prefix) {
                            groupIndices.push(idx);
                        }
                    });

                    if (groupIndices.length === 0) continue;

                    // --- ШАГ 1: УДАЛЕНИЕ ПУСТЫХ ХВОСТОВ В ГРУППЕ ---
                    for (let i = groupIndices.length - 1; i > 0; i--) {
                        const realIndex = groupIndices[i];
                        const input = this.inputs[realIndex];

                        if (input && !input.link) {
                            this.removeInput(realIndex);
                            structureChanged = true;
                            
                            // Корректируем индексы в локальном массиве после удаления слота из ноды
                            groupIndices.splice(i, 1);
                            for (let j = i; j < groupIndices.length; j++) {
                                groupIndices[j]--;
                            }
                        } else {
                            break; 
                        }
                    }

                    // --- ШАГ 2: АВТОМАТИЧЕСКОЕ ДОБАВЛЕНИЕ НОВОГО СЛОТА ---
                    // Проверяем самый последний оставшийся слот в этой группе
                    const lastRealIndex = groupIndices[groupIndices.length - 1];
                    const lastInput = this.inputs[lastRealIndex];

                    // Если последний слот группы СЕЙЧАС подключен — создаем под него новый
                    if (lastInput && lastInput.link) {
                        const totalForPrefix = groupIndices.length;
                        
                        const newName = totalForPrefix === 0 ? prefix : `${prefix}${totalForPrefix + 1}`;
                        const inputType = inputTypes[prefix] || "IMAGE";

                        if (inputType === "STRING") {
                            this.addInput(newName, inputType, { multiline: true, default: "", forceInput: true });
                        } else {
                            this.addInput(newName, inputType);
                        }
                        structureChanged = true;
                    }
                }

                if (structureChanged) {
                    this.setSize(this.computeSize());
                    this.setDirtyCanvas(true, true);
                }
            }, 1);
        };
    }
});