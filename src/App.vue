<template>
  <div class="app">
    <RouterView></RouterView>
    <!-- <Login/> -->
  </div>
</template>

<script lang="ts">
  import Login from './components/Login.vue'
  import { RouterView } from 'vue-router';
  export default {
    name: "App",
    components:{Login}
  }

</script>

<style>
  .app {
  display: flex;
  justify-content: center; /* 水平居中 */
  align-items: center;     /* 垂直居中 */
  height: 100vh;           /* 使容器占满视口高度 */
  }
</style>