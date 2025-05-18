class AuthRouter:
    AUTH_APPS = ['accounts', 'admin', 'auth', 'contenttypes', 'sessions']

    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.AUTH_APPS:
            return 'user_management_db'
        return 'default'

    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.AUTH_APPS:
            return 'user_management_db'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        return True

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in self.AUTH_APPS:
            return db == 'user_management_db'
        return db == 'default'