def RMSprop_update(params,
                   grads,
                   square_avgs,
                   weight_decay,
                   lr,
                   eps,
                   alpha):
    """Functional API that performs rmsprop algorithm computation.
    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        avg = square_avg.sqrt().add_(eps)
        param.addcdiv_(grad, avg, value=-lr)
    return params


def test_model(env, model):
    '''
    Test a model for the total rewards it can collect
    '''
    observation = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = model.predict(observation)

        next_observation, reward, done, _ = env.step(action)

        observation = next_observation

        total_reward += reward

    return total_reward
