import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-mono font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary/15 text-primary",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        destructive: "border-transparent bg-destructive/20 text-destructive",
        outline: "border-border text-muted-foreground",
        long:   "border-green-500/20  bg-green-500/10  text-green-400",
        short:  "border-red-500/20   bg-red-500/10   text-red-400",
        exit:   "border-amber-500/20 bg-amber-500/10 text-amber-400",
        none:   "border-border       bg-transparent   text-muted-foreground",
        HIGH:   "border-green-500/20  bg-green-500/10  text-green-400",
        MEDIUM: "border-amber-500/20 bg-amber-500/10 text-amber-400",
        LOW:    "border-orange-500/20 bg-orange-500/10 text-orange-400",
        NOT_OU: "border-red-500/20   bg-red-500/10   text-red-400",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
