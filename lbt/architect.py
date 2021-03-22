import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])



def cusloss(inp, tar):
    m = nn.Softmax(-1)
    lm = nn.LogSoftmax(-1)
    lenn = inp.shape[0]
    inp = lm(inp)
    tar = m(tar)
    out = inp*tar


    ll = (out.sum()*(-1))/lenn
    return ll


class Architect(object):

  def __init__(self, model, student, momentum=0.9, weight_decay=3e-4, lambda_par=1.0, arch_learning_rate = 3e-4, arch_weight_decay=1e-3):
    self.network_momentum = momentum
    self.network_weight_decay = weight_decay
    self.model = model
    self.student = student
    self.lambda_par = lambda_par
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=arch_learning_rate, betas=(0.5, 0.999), weight_decay=arch_weight_decay)


  def _concat_mgrads(self, xs):
    xss = []


    for v, g in zip(self.model.parameters(), xs):
      if g is None:
        xss.append(torch.zeros_like(v).view(-1))
      else:
        xss.append(g.view(-1))      

    return torch.cat(xss)


  def _concat_sgrads(self, xs):
    xss = []
    for v, g in zip(self.student.parameters(), xs):
      if g is None:
        xss.append(torch.zeros_like(v).view(-1))
      else:
        xss.append(g.view(-1))      

    return torch.cat(xss)


  def _concat_hessian(self, xs, unrolled_model):

    xss = []
    for v, g in zip(unrolled_model.parameters(), xs):
      if g is None:
        xss.append(torch.zeros_like(v).view(-1))
      else:
        xss.append(g.view(-1))      

    return torch.cat(xss)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = self._concat_mgrads(torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def _compute_unrolled_model1(self, input, target, eta, unrolled_model, input_unlabeled, student_optimizer):
    loss1 = self.student._loss(input, target)
    l1 = self.model(input_unlabeled)
    logits1 = self.student(input_unlabeled)
    crit = nn.MSELoss()
    loss2 = crit(logits1, l1)
    loss = loss1+(self.lambda_par*loss2)
    theta = _concat(self.student.parameters()).data
    try:
      moment = _concat(student_optimizer.state[v]['momentum_buffer'] for v in self.student.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = self._concat_sgrads(torch.autograd.grad(loss, self.student.parameters(), allow_unused=True)).data + self.network_weight_decay*theta
    unrolled_student = self._construct_model_from_theta1(theta.sub(eta, moment+dtheta))
    return unrolled_student

  def step(self, input_train, target_train, input_valid, target_valid, input_unlabeled, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()
    
  def step1(self, input_train, target_train, input_valid, target_valid, input_unlabeled, eta, network_optimizer, student_optimizer, unrolled):
    self.optimizer.zero_grad()
    
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_student = self._compute_unrolled_model1(input_train, target_train, eta, unrolled_model, input_unlabeled, student_optimizer)
    
    unrolled_stud_loss = unrolled_student._loss(input_valid, target_valid)
    unrolled_stud_loss.backward()
    
    vector_s_dash = [v.grad.data for v in unrolled_student.parameters()]
    
    implicit_grads = self._outer1(vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, unrolled_student, eta)
    
    for v, g in zip(self.model.arch_parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data, requires_grad=True)
      else:
        v.grad.data.copy_(g.data)
    
    self.optimizer.step()
    

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()

    vector = []
    for v in unrolled_model.parameters():
      try:
        vector.append(v.grad.data)
      except:
        vector.append(torch.zeros_like(v))

    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data, requires_grad=True)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict, strict=False)
    return model_new.cuda()

  def _construct_model_from_theta1(self, theta):
    model_new = self.student.new()
    model_dict = self.student.state_dict()

    params, offset = {}, 0
    for k, v in self.student.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict, strict=False)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, R = None, r=1e-2):
    if R is None:
      R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):      
      if v is None:
        p.data.add_(R, torch.zeros_like(p))
      else:
        p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      if v is None:
        p.data.add_(2*R, torch.zeros_like(p))
      else:
        p.data.sub_(2*R, v)
      
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      if v is None:
        p.data.add_(R, torch.zeros_like(p))
      else:
        p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def _outer1(self, vector_s_dash, input_train, target_train, input_unlabeled, unrolled_model, unrolled_student, eta, r=1e-2):
    R1 = r / self._concat_hessian(vector_s_dash, unrolled_student).norm()
    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss1 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss1, unrolled_model.parameters(), allow_unused=True)
    R_1 = r / self._concat_hessian(vector_t_dash, unrolled_model).norm()
    grad_part1 = self._hessian_vector_product(vector_t_dash, input_train, target_train, R=R_1)
    

    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.sub_(2*R1, v)
    logits1 = self.student(input_unlabeled)
    logits2 = unrolled_model(input_unlabeled)
    loss2 = cusloss(logits1, logits2)
    
    vector_t_dash = torch.autograd.grad(loss2, unrolled_model.parameters(), allow_unused=True)
    R_1 = r / self._concat_hessian(vector_t_dash, unrolled_model).norm()
    grad_part2 = self._hessian_vector_product(vector_t_dash, input_train, target_train,  R = R_1)

    for p, v in zip(self.student.parameters(), vector_s_dash):
      p.data.add_(R1, v)

    return [(x-y).div_((2*R1)/(eta*eta*self.lambda_par)) for x, y in zip(grad_part1, grad_part2)]


