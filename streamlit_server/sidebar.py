import streamlit as st


def draw_sidebar(state):
    st.sidebar.header('Customise plots')
    all_draw_params = []

    for i, s in enumerate(state.schedulers):
        draw_params = {}
        scheduler_name = list(s.keys())[0]
        new_scheduler = st.sidebar.selectbox(
            'Select type of lr scheduler',
            state.sched_builder.supported_schedulers,
            index=state.sched_builder.supported_schedulers.index(scheduler_name),
            key=str(i+1)
        )

        if new_scheduler != scheduler_name:
            params = state.sched_builder.schedulers_params[new_scheduler]
        else:
            params = s[scheduler_name]

        for param in params:
            if (param['type'] == 'float') or (param['type'] == 'int'):
                p = st.sidebar.slider(
                    label=param['param'],
                    min_value=param['min'],
                    max_value=param['max'],
                    value=param['default'],
                    key=param['param'] + '_slider_' + str(i)
                )

                draw_params[param['param']] = p

        draw_params['name'] = new_scheduler

        if st.sidebar.checkbox('Add to plot?', value=True, key='bool' + str(i)):
            all_draw_params.append(draw_params)

            if new_scheduler == 'ReduceLROnPlateau':
                st.warning('ReduceLROnPlateau requires loss tracking. We simulate only constant loss behaviour! ')

        st.sidebar.text('-_' * 18)

    history = []
    for s in all_draw_params:
        name = s['name']
        p = s.copy()
        del p['name']
        scheduler = state.sched_builder.get_scheduler(name, p)
        scheduler.iterate(99)
        history.append({
            'name': name,
            'lr': scheduler.lr_history
        })

    return history, all_draw_params













