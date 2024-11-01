from homegrid.__init__ import HomeGrid


def make_env(
    need_reset=True,
    gpt_pool=False,
    train_ratio=None,
    mode=None,
    val_ratio=None,
    disturb_hind=False,
    disturb_fore=False,
):
    env = HomeGrid(
        lang_types=["task", "dynamics", "corrections", "future"],
        gpt_pool=gpt_pool,
        train_ratio=train_ratio,
        mode=mode,
        val_ratio=val_ratio,
        disturb_hind=disturb_hind,
        disturb_fore=disturb_fore,
    )
    if need_reset:
        env.reset()
    return env
