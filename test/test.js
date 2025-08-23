const request = require('supertest');
const app = require('../server');

describe('Cincinnatus Life Tracker API', () => {
  describe('Health Check', () => {
    it('should return health status', async () => {
      const res = await request(app)
        .get('/health')
        .expect(200);
      
      expect(res.body.status).toBe('OK');
      expect(res.body.message).toContain('Cincinnatus Life Tracker Server is running');
    });
  });

  describe('Authentication', () => {
    it('should register a new user', async () => {
      const userData = {
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        password: 'TestPassword123!'
      };

      const res = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(res.body.success).toBe(true);
      expect(res.body.data.firstName).toBe(userData.firstName);
      expect(res.body.data.email).toBe(userData.email);
      expect(res.body.data.token).toBeDefined();
    });

    it('should login existing user', async () => {
      const loginData = {
        email: 'john.doe@example.com',
        password: 'TestPassword123!'
      };

      const res = await request(app)
        .post('/api/auth/login')
        .send(loginData)
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.data.token).toBeDefined();
    });
  });

  describe('Prompts', () => {
    it('should get featured prompts', async () => {
      const res = await request(app)
        .get('/api/prompts/featured')
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should get prompts by category', async () => {
      const res = await request(app)
        .get('/api/prompts?category=business')
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(Array.isArray(res.body.data)).toBe(true);
    });
  });
});
