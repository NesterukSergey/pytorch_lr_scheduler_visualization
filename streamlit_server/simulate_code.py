def simulate_code(state, p):
    init = '''
    model = torch.nn.Linear(1, 1)  # dummy model
    optimizer = torch.optim.Adam(model.parameters())  # dummy optimizer
    '''

    name = p['name']
    del p['name']
    params = state.sched_builder.parse_params(p)

    scheduler = 'scheduler = torch.optim.lr_scheduler.{}(optimizer, {})'.format(name, params)

    iterate = '''
    
    for i in range(epochs):
        optimizer.zero_grad()
        train(...)
        validate(...)
        loss.backward()
        optimizer.step()
    '''

    if name == 'ReduceLROnPlateau':
        step = '    scheduler.step(loss)'
    else:
        step = '    scheduler.step()'

    return init + scheduler + iterate + step
