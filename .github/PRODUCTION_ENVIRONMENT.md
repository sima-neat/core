# Vulcan production GitHub environment

Vulcan production jobs use the GitHub environment `prod`. Legacy `production`
inputs remain accepted; only the job environment is normalized. Runtime values,
AWS resource names, and runner labels such as `vulcan-production` are unchanged.

Keep production environment variables, secrets, and protection rules on `prod`.
Retain the old environment and both AWS trust aliases until active caller branches
have migrated and real GitHub OIDC authentication has been verified.

Tracked in https://github.com/sima-neat/vulcan/issues/259.
