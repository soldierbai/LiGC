<template>
  <div class="login-container">
    <div class="form-wrapper">
      <h2 class="title">登录 / Login</h2>
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <label for="username">用户名</label>
          <input v-model="loginForm.username" type="text" required />
        </div>
        <div class="form-group">
          <label for="password">密码</label>
          <input v-model="loginForm.password" type="password" required />
        </div>
        <button type="submit" :disabled="!isValid">登录</button>
      </form>
      <div class="footer">
        <!-- 第一个链接：添加类名 "hover-effect" -->
        <router-link to="/forgot-password" class="hover-effect">忘记密码?</router-link>
        <span>·</span>
        <!-- 第三个链接：添加类名 "hover-effect" -->
        <router-link to="/register" class="hover-effect">注册账号</router-link>
      </div>
    </div>
    <!-- <p v-if="error" class="error">{{ error }}</p> -->
  </div>
</template>
    
<script setup lang="ts" name="Login">
  import { reactive, computed } from 'vue';
  import { useRouter } from 'vue-router';
  
  const loginForm = reactive({
    username: '',
    password: '',
  });
  
  const isValid = computed(() => loginForm.username && loginForm.password);
  const router = useRouter();
  
  const handleLogin = async () => {
    try {
      // 模拟登录请求
      await new Promise(resolve => setTimeout(resolve, 1000));
      router.push('/dashboard');
    } catch (error) {
      console.error('登录失败:', error);
    }
  };
</script>
    
<style scoped>
  .login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background: linear-gradient(135deg, #ffdfe1d2, #bcd7ff);
    padding: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    width: 100%;
  }

  .form-wrapper {
    background: white;
    padding: 3rem;
    border-radius: 12px;
    width: 400px;
    text-align: center;
  }

  .title {
    margin-bottom: 2rem;
    color: #333;
    font-size: 2.2rem;
  }

  .form-group {
    margin-bottom: 1.5rem;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    color: #666;
  }

  input {
    width: 80%;
    padding: 1.2rem;
    border: 2px solid #e0f7fa;
    border-radius: 8px;
    transition: border-color 0.3s ease;
  }

  input:focus {
    border-color: #42b983;
    outline: none;
  }

  button {
    width: 100%;
    padding: 1.2rem;
    background: #42b983;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1.2rem;
    cursor: pointer;
    transition: background 0.3s ease;
  }

  button:hover {
    background: #38a175;
  }

  button:disabled {
    background: #b3b3b3;
    cursor: not-allowed;
  }

  .footer {
    margin-top: 2rem;
    font-size: 1rem;
    color: #999;
  }

  .footer a {
    color: #999;
    margin: 0 1rem;
    text-decoration: none;
  }

  .footer span {
    margin: 0 1rem;
  }

  .hover-effect {
  text-decoration: none;
  color: #999; /* 原始颜色（根据主题调整） */
  transition: color 0.3s ease, text-decoration 0.3s ease;
}

  /* 悬停时的状态 */
  .hover-effect:hover {
    text-decoration: underline;
    color: #444; /* 颜色加深（根据主题调整） */
    cursor: pointer;
  }
</style>