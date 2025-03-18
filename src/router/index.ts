import { createRouter,createWebHistory } from "vue-router"
import Login from '@/components/Login.vue'
import Register from "@/components/Register.vue"
import Dashboard from "@/components/Dashboard.vue"
const router = createRouter({
    history:createWebHistory(),
    routes:[
        {
            path:'/',
            component:Login
        },

        {
            path:'/login',
            component:Login
        },
        {
            path:'/register',
            component:Register
        },
        {
            path:'/dashboard',
            component:Dashboard
        }
    ]
})


export default router