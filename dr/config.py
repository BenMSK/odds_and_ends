import math

DEG2RAD = math.pi / 180.0
class Vehicle:
    def __init__(self, steering_ratio = 15.56, x=0, y=0, velocity = 0, steering = 0):
        self.steering_ratio = steering_ratio#steering / yaw #14.60 good
        self.x = x
        self.y = y
        self.velocity = velocity
        self.steering = steering#rad
        self.yaw = self.steering/self.steering_ratio
        self.wheel_base = 2.845

    def CTRV(self, current_steering, current_velocity, dt, forward):
        current_yaw = DEG2RAD*current_steering/self.steering_ratio
        yaw_rate = (current_yaw - self.yaw)/dt
        if forward:
            sign = +1
        else:
            sign = -1
        if yaw_rate != 0:
            self.x += ((sign)*self.velocity/yaw_rate)*(math.sin(self.yaw + yaw_rate*dt)
                                                -math.sin(self.yaw))
            self.y -= ((sign)*self.velocity/yaw_rate)*(-math.cos(self.yaw + yaw_rate*dt)
                                                +math.cos(self.yaw))
            self.yaw += yaw_rate*dt
            self.velocity = current_velocity
        else:#dt ==0
            self.x += (sign)*self.velocity*math.cos(current_yaw)*dt
            self.y -= (sign)*self.velocity*math.sin(current_yaw)*dt
            self.yaw += yaw_rate*dt
            self.velocity = current_velocity


    def CTRA(self, yaw_rate, current_accel, dt, forward):
        if forward:
            sign = +1
        else:
            sign = -1

        self.x += (1/((self.yaw + yaw_rate*dt)**2)) * (((sign)*self.velocity*yaw_rate + current_accel*yaw_rate*dt)*math.sin(self.yaw + yaw_rate*dt)
                                           + current_accel*math.cos(self.yaw + yaw_rate*dt)
                                           - (sign)*self.velocity*yaw_rate*math.sin(self.yaw + yaw_rate*dt)
                                           - current_accel*math.cos(self.yaw + yaw_rate*dt))

        self.y += (1/((self.yaw + yaw_rate*dt)**2)) * ((-(sign)*self.velocity*yaw_rate - current_accel*yaw_rate*dt)*math.cos(self.yaw + yaw_rate*dt)
                                           + current_accel*math.sin(self.yaw + yaw_rate*dt)
                                           + (sign)*self.velocity*yaw_rate*math.cos(self.yaw + yaw_rate*dt)
                                           - current_accel*math.sin(self.yaw + yaw_rate*dt))

        self.yaw += yaw_rate*dt
        self.velocity += current_accel*dt

    def BicycleModel(self, current_steering, current_velocity, dt, forward, alpha = 0.5):
        current_velocity*= 1.2
        current_velocity = alpha*current_velocity+(1-alpha)*self.velocity# control(?)
        if forward:
            sign = +1.
        else:
            sign = -1.
        
        current_yaw = DEG2RAD*current_steering/self.steering_ratio#theta(t)
        slip_beta = math.atan2(math.tan(current_yaw), 2)
        self.x += (sign)*current_velocity*math.cos(self.yaw + slip_beta)*dt
        self.y -= (sign)*current_velocity*math.sin(self.yaw + slip_beta)*dt
        self.yaw += dt*(sign)*current_velocity*math.tan(current_yaw)*math.cos(slip_beta) / self.wheel_base
        
        self.velocity = current_velocity
        