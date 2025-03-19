<template>
  <div class="chat-container">
    <!-- 左侧历史记录 -->
    <div class="history-panel">
      <div class="header">
        <h2 class="title">对话历史</h2>
        <n-button 
          type="primary" 
          class="new-chat-btn"
          @click="addNewChat"
        >
          <template #icon>
            <n-icon><PlusIcon /></n-icon>
          </template>
          新建对话
        </n-button>
      </div>
      <div class="history-list">
        <div 
          v-for="(item, index) in mockHistory" 
          :key="item.id"
          class="history-item"
          :class="{ active: activeIndex === index }"
          @click="setActiveChat(index)"
        >
          <div class="item-header">
            <n-ellipsis class="title">{{ item.title }}</n-ellipsis>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <n-tag size="small" :type="item.type">{{ item.model }}</n-tag>
          </div>
          <div class="item-footer">
            <n-time :time="item.time" type="relative" class="time-element" />
            <n-icon 
              size="25" 
              class="action-icon" 
              @click.stop="deleteHistory(item.id)"
            >
              <DeleteIcon />
            </n-icon>
          </div>
        </div>
      </div>
    </div>
  
    <!-- 右侧聊天区 -->
    <div class="chat-main">
      <div class="message-list">
        <div 
          v-for="(msg, index) in mockMessages" 
          :key="index"
          class="message-bubble"
          :class="{ 'user-msg': msg.role === 'user' }"
        >
          <div class="content">
            <div v-if="msg.role === 'user'" class="text">{{ msg.content }}</div>
            <div v-else>
              <div class="black-box">
                <div v-if="msg.thinking_content" v-html="formatMarkdown(msg.thinking_content)"></div>
              </div>
              <div v-if="msg.content" v-html="formatMarkdown(msg.content)" class="text"></div>
            </div>
          </div>
        </div>
      </div>

      <div class="input-area">
        <div class="input-wrapper">
          <n-input
            v-model:value="inputText"
            type="textarea"
            placeholder="输入您的问题..."
            :autosize="{ minRows: 1 }"
            @keydown.enter.prevent="sendMessage"
          />
          <n-button
            type="primary"
            class="send-btn"
            :disabled="!inputText || outputting"
            @click="sendMessage"
          >
            发送
            <template #icon>
              <n-icon><SendIcon /></n-icon>
            </template>
          </n-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { 
  NButton, 
  NEllipsis, 
  NTag, 
  NTime, 
  NIcon, 
  NInput
} from 'naive-ui'
import { 
  AddOutline as PlusIcon,
  PaperPlaneOutline as SendIcon,
  CopyOutline as CopyIcon,
  EllipsisHorizontal as MoreIcon,
  TrashOutline as DeleteIcon
} from '@vicons/ionicons5'
import MarkdownIt from 'markdown-it'

interface ChatHistory {
  id: string
  title: string
  model: string
  type: 'success' | 'info' | 'warning' | 'error'
  time: number
  messages: ChatMessage[] // 包含对话历史的消息
}

interface ChatMessage {
  content: string
  thinking_content: string
  role: 'user' | 'assistant'
}

// 响应式数据
const inputText = ref('')
const activeIndex = ref(0)
const outputting = ref(false)

// 生成随机ID
const generateRandomId = (): string => {
  return Math.random().toString(36).substring(2, 8) + '-' + Date.now()
}

// 历史记录数据
const mockHistory = ref<ChatHistory[]>([])

// 消息记录
const mockMessages = ref<ChatMessage[]>([])

// Markdown解析器
const md = new MarkdownIt()

// 格式化Markdown内容
const formatMarkdown = (content: string): string => {
  // 如果内容包含<think>...</think>标签，提取标签内的内容
  if (content.includes('<think>') && content.includes('</think>')) {
    const start = content.indexOf('<think>') + '<think>'.length;
    const end = content.indexOf('</think>');
    content = content.substring(start, end);
  }
  return md.render(content);
}
// 设置当前激活的对话
const setActiveChat = (index: number) => {
  activeIndex.value = index
  mockMessages.value = mockHistory.value[index].messages // 更新右侧消息列表
}

// 新建对话
const addNewChat = () => {
  const newChat: ChatHistory = {
    id: generateRandomId(),
    title: `对话 ${mockHistory.value.length + 1}`,
    model: 'Deepseek',
    type: 'success',
    time: Date.now(),
    messages: []
  }
  
  mockHistory.value.push(newChat)
  activeIndex.value = mockHistory.value.length - 1
  mockMessages.value = [] // 清空当前消息
}

// 删除对话
const deleteHistory = (id: string) => {
  const index = mockHistory.value.findIndex(item => item.id === id)
  if (index !== -1) {
    mockHistory.value.splice(index, 1)
    if (activeIndex.value >= mockHistory.value.length) {
      activeIndex.value = mockHistory.value.length - 1
    }
    mockMessages.value = [] // 清空当前消息
  }
}

// 修改后的sendMessage函数
const sendMessage = async () => {
  if (!inputText.value.trim()) return

  outputting.value = true
  // 创建用户消息
  const userMessage: ChatMessage = {
    content: inputText.value,
    thinking_content: '',
    role: 'user'
  }
  
  // 创建初始助手消息
  const assistantMessage: ChatMessage = {
    content: '',
    thinking_content: '',
    role: 'assistant'
  }
  

  const history_messages = mockMessages.value
  

  // 更新消息列表
  mockMessages.value.push(userMessage)
  mockMessages.value.push(assistantMessage)
    
  // 清空输入
  inputText.value = ''
  console.log(mockMessages.value)

  try {
    
    const response = await fetch('http://127.0.0.1:5010/api/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: userMessage.content,
        messages: history_messages
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) return

    const assistantIndex = mockMessages.value.length - 1

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      // 解码数据块
      const chunk = new TextDecoder().decode(value)
      try {
        // 处理SSE格式数据
        chunk.split('\n\n').forEach(event => {
          const match = event.match(/data:\s*(\{.*\})/)
          if (match) {
            
            const data = JSON.parse(match[1])
            console.log(data.message)
            if (data.message.content) {
              // 拼接内容
              mockMessages.value[assistantIndex].content += data.message.content
              // 触发视图更新
              mockMessages.value = [...mockMessages.value]
            }
            else if (data.message.thinking_content) {
              // 拼接内容
              mockMessages.value[assistantIndex].thinking_content += data.message.thinking_content
              // 触发视图更新
              mockMessages.value = [...mockMessages.value]
            }
          }
        })
      } catch (e) {
        console.error('JSON解析错误:', e)
        mockMessages.value[assistantIndex].content += '\n[解析错误]'
        outputting.value = false
        return
      }
    }
  } catch (error) {
    const assistantIndex = mockMessages.value.length - 1
    console.error('请求失败:', error)
    mockMessages.value[assistantIndex].content += '\n[请求失败，请检查接口]'
    outputting.value = false
    return
  }
  outputting.value = false
}
</script>

<style scoped>
.chat-container {
    display: flex;
    height: 100vh;
    width: 100%;
    background: linear-gradient(135deg, #f8f5ff 0%, #e3f2fd 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* 左侧历史面板 - 参考文献2的卡片式设计 */
.history-panel {
    display: flex;
    flex-direction: column;
    padding: 2rem;
    width: 280px;
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 4px 0 15px rgba(0, 0, 0, 0.05);
    margin: 1rem;
    border-radius: 0 20px 20px 0;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(220, 220, 220, 0.3);
    
    /* 新增滚动条美化 */
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #42b983 transparent;
}

.history-panel::-webkit-scrollbar {
    width: 6px;
}

.history-panel::-webkit-scrollbar-thumb {
    background-color: #42b983;
    border-radius: 3px;
}

.header {
    padding: 1rem;
    border-bottom: 1px solid #eee;
    margin-bottom: 1rem;
    
    /* 参考文献7的按钮样式优化 */
    n-button {
        background: linear-gradient(135deg, #42b983 0%, #38a175 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 2px 6px rgba(66, 185, 131, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        
        &:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(66, 185, 131, 0.4);
        }
    }
}

/* 右侧聊天主区域 - 参考文献6的布局优化 */
.chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border-radius: 20px 0 0 20px;
    margin: 1rem;
    box-shadow: 
        -4px 0 15px rgba(0, 0, 0, 0.05),
        inset 1px 0 0 rgba(255, 255, 255, 0.1);
}

/* 消息列表容器 - 参考文献3的滚动优化 */
.message-list {
    flex: 1;
    overflow-y: auto;
    padding-right: 1rem;
    margin-bottom: 1.5rem;
    
    /* 自定义滚动条 */
    &::-webkit-scrollbar {
        width: 8px;
    }
    
    &::-webkit-scrollbar-thumb {
        background: rgba(200, 200, 200, 0.6);
        border-radius: 4px;
    }
}

/* 消息气泡 - 参考文献2的阴影优化 */
.message-bubble {
    max-width: 75%;
    margin: 1rem 0;
    padding: 1.2rem 1.5rem;
    background: white;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease;
    
    &.user-msg {
        margin-left: auto;
        background: linear-gradient(135deg, #42b98310 0%, #38a17515 100%);
        border: 1px solid #42b98320;
    }
    
    &:hover {
        transform: translateX(4px);
    }
}

.input-area {
  position: relative;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(220, 220, 220, 0.2);
  padding: 1rem;
  display: flex;
  align-items: center;
}

.input-wrapper {
  display: flex;
  width: 100%;
  gap: 1rem;
}

input-area .n-input {
  flex: 1 1 auto;
  border: none !important;
  border-color: transparent !important;
  outline: none !important;
  padding: 0.8rem;
  background: transparent;
}


input-area .send-btn {
  flex-shrink: 0; /* 防止按钮被压缩 */
  padding: 0.6rem 1.8rem;
  justify-content: center; /* 水平居中按钮内容 */
  background: #007bff; /* 建议添加基础按钮颜色 */
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: background 0.2s; /* 添加交互反馈 */
}


input-area .send-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 3px 8px rgba(66, 185, 131, 0.4);
}


input-area .send-btn:disabled {
  background: #e0e0e0;
  transform: none;
  box-shadow: none;
}


/* 新增动画效果 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}


.message-bubble {
    animation: fadeIn 0.3s ease-out;
}


/* 实时更新时的闪烁效果 */
.message-bubble:last-child {
  animation: blink 1s ease-in-out infinite;
}

@keyframes blink {
  0% { background-color: rgba(245,245,245,0.1); }
  50% { background-color: rgba(245,245,245,0.3); }
  100% { background-color: rgba(245,245,245,0.1); }
}

/* 响应式优化 */
@media (max-width: 768px) {
    .history-panel {
        width: 240px;
        border-radius: 0;
    }
    
    .chat-main {
        margin: 0;
        border-radius: 0;
    }
}


.item-footer {
  display: flex;                  
  justify-content: space-between; 
  align-items: center;            
  padding: 8px 16px;              
}


.time-element {
  font-size: 14px;
  color: #666;
}


.action-icon {
  cursor: pointer;          
  transition: all 0.3s ease;
}


.action-icon:hover {
  transform: scale(1.1);   
  color: rgb(223, 81, 81);
}

.black-box {
  color: rgb(79, 79, 79);
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 0px;
  padding-bottom: 0px;
  margin: 10px;
  margin-left: 0px;
  
  border-left: 4px solid rgba(163, 163, 163, 0.5);
  font-size:13px;
}

</style>