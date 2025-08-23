import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';

const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,

      // Initialize auth state
      initialize: async () => {
        try {
          const token = localStorage.getItem('token');
          if (token) {
            api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            const response = await api.get('/api/users/profile');
            set({
              user: response.data.data,
              token,
              isAuthenticated: true,
              isLoading: false,
            });
          } else {
            set({ isLoading: false });
          }
        } catch (error) {
          console.error('Auth initialization error:', error);
          localStorage.removeItem('token');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },

      // Login
      login: async (email, password) => {
        try {
          const response = await api.post('/api/auth/login', {
            email,
            password,
          });
          
          const { token, user } = response.data.data;
          
          localStorage.setItem('token', token);
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          return { success: true };
        } catch (error) {
          return {
            success: false,
            error: error.response?.data?.message || 'Login failed',
          };
        }
      },

      // Register
      register: async (userData) => {
        try {
          const response = await api.post('/api/auth/register', userData);
          
          const { token, user } = response.data.data;
          
          localStorage.setItem('token', token);
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
          
          return { success: true };
        } catch (error) {
          return {
            success: false,
            error: error.response?.data?.message || 'Registration failed',
          };
        }
      },

      // Logout
      logout: () => {
        localStorage.removeItem('token');
        delete api.defaults.headers.common['Authorization'];
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      // Update user profile
      updateProfile: async (profileData) => {
        try {
          const response = await api.put('/api/users/profile', profileData);
          const updatedUser = response.data.data;
          
          set({ user: updatedUser });
          return { success: true, user: updatedUser };
        } catch (error) {
          return {
            success: false,
            error: error.response?.data?.message || 'Profile update failed',
          };
        }
      },

      // Refresh user data
      refreshUser: async () => {
        try {
          const response = await api.get('/api/users/profile');
          const updatedUser = response.data.data;
          
          set({ user: updatedUser });
          return { success: true, user: updatedUser };
        } catch (error) {
          console.error('User refresh error:', error);
          return { success: false };
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
);

export default useAuthStore;
