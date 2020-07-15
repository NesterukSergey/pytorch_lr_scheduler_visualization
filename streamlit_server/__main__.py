from sidebar import *
from plot import *
from State import *
from simulate_code import *

st.title('Pytorch lr schedulers visualization')
state = State()

logscale = st.checkbox('Use logscale?', value=state.logscale)
state.logscale = logscale

history, all_draw_params = draw_sidebar(state)
plot(state, history)

for p in all_draw_params:
    st.code(simulate_code(state, p))

