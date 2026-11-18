from hypo_release.paths import training_config
CONFIG=training_config()
CONFIG['training'].update(epochs=3000,early_stop_patience=20,snapshot_epochs=[])
