<template>
    <div class="login-container">
      <div class="form-wrapper">
        <h2 class="title">注册 / Register</h2>
        <form @submit.prevent="handleRegister">
          <div class="form-group">
            <label for="username">用户名</label>
            <input v-model="registerForm.username" type="text" required />
          </div>
          <div class="form-group">
            <label for="password">密码</label>
            <input v-model="registerForm.password" type="password" required />
          </div>
          <!-- 新增确认密码字段 -->
        <div class="form-group">
          <label for="confirmPassword">确认密码</label>
          <input v-model="registerForm.confirmPassword" type="password" required />
          <p v-if="!isPasswordMatch" class="error-message">密码不匹配</p>
        </div>

          <button type="submit" :disabled="!isValid">注册</button>
        </form>
        <div class="footer">
          <router-link to="/login">已有账号？登录</router-link>
        </div>
      </div>
      <!-- <p v-if="error" class="error">{{ error }}</p> -->
    </div>
</template>
  
<script setup lang="ts" name="Register">
  import { reactive, computed } from 'vue';
  import { useRouter } from 'vue-router'; // 引入路由模块

  const router = useRouter(); // 获取路由实例

  const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '', // 新增字段
  });

  // 更新验证逻辑
  const isValid = computed(() => {
    return (
      registerForm.username &&
      registerForm.password &&
      registerForm.confirmPassword === registerForm.password
      );
    });

    // 实时验证密码是否匹配
  const isPasswordMatch = computed(() => 
    registerForm.password === registerForm.confirmPassword
  );

  // 检查所有必填项是否已填写
  const areAllFieldsFilled = computed(() => {
    return registerForm.username.trim() !== '' &&
          registerForm.password.trim() !== '' &&
          registerForm.confirmPassword.trim() !== '';
  });

  const handleRegister = async () => {
    // 提交前最终验证
    if (!isPasswordMatch.value) {
      alert('密码不匹配，请重新输入！');
      return;
    }

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      // 注册成功后跳转登录页
      router.push('/login');
    } catch (error) {
      console.error('注册失败:', error);
      alert('注册失败，请稍后重试！');
    }
  };
</script>

<style scoped>
  .login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background: linear-gradient(135deg, #ffe88dd2, #bcd7ff);
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
    color: #42b983;
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

  .error-message {
  color: red;
  font-size: 0.9em;
  margin-top: 5px;
  }
</style>